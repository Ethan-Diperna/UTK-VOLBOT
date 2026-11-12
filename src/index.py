"""Build both search indexes from the same chunks with one command.

  python -m src.index                                  # chunk, embed, index
  python -m src.index --strategy packed --max-words 300

Two indexes, because they fail in different ways:

* Vector index (Chroma): each chunk is turned into a 384-number embedding by a
  small local model. Chunks whose meaning is close to the question are found
  even when the wording differs ("classes about hacking" finds cybersecurity).
  Weak on exact identifiers: "COSC 302" and "COSC 312" look almost the same.
* Keyword index (BM25): scores chunks by the question's words, weighting rare
  words more. Strong on exact course codes, blind to synonyms.

Phase 4 combines the two. Both are rebuilt from scratch on every run so they
can never disagree with chunks.jsonl.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from rank_bm25 import BM25Okapi

from src import config as cfg
from src.chunk import STRATEGIES, chunk_docs, load_jsonl, save_jsonl, stats

# A course code is 2-5 letters, an optional space or hyphen, then 3 digits.
# Text is lowercased first, so this sees "cosc 302", "cosc302", "cosc-302".
CODE_RE = re.compile(r"\b([a-z]{2,5})[\s-]?(\d{3})\b")
TOKEN_RE = re.compile(r"[a-z0-9]+")
# Chroma metadata values must be plain strings/numbers, so only these go in.
METADATA_FIELDS = ("doc_id", "url", "title", "section", "doc_type", "dept", "code", "catalog_year")
EMBED_BATCH_SIZE = 32


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, with every course code written three ways.

    "COSC 302", "COSC302" and "cosc 302" all become ["cosc", "302", "cosc302"].
    The joined token makes an exact code match count as one rare, heavily
    weighted word; the separate parts keep "which COSC classes ..." working.
    The same function is used on chunks and on questions, so they always agree.
    A stray match such as "is 302" only adds a harmless extra token."""
    text = CODE_RE.sub(r"\1 \2 \1\2", text.lower())
    return TOKEN_RE.findall(text)


class Bm25Index:
    """BM25 over tokenized chunks. Stored as plain JSON (ids + tokens) and
    rebuilt in memory on load, which takes milliseconds at this size."""

    def __init__(self, ids: list[str], tokens: list[list[str]], meta: dict | None = None):
        self.ids = ids
        self.tokens = tokens
        self.meta = meta or {}
        self._bm25 = BM25Okapi(tokens)

    @classmethod
    def build(cls, chunks: list[dict]) -> "Bm25Index":
        return cls([c["id"] for c in chunks], [tokenize(c["text"]) for c in chunks])

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Top-k (chunk id, score), best first. Chunks sharing no word with the
        query score 0 and are left out."""
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.ids, scores), key=lambda pair: pair[1], reverse=True)
        return [(cid, float(score)) for cid, score in ranked[:k] if score > 0]

    def save(self, path: Path, meta: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"ids": self.ids, "tokens": self.tokens, "meta": meta or self.meta}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Bm25Index":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload["ids"], payload["tokens"], payload.get("meta"))


def load_embedder(model_name: str = cfg.EMBEDDING_MODEL):
    # Imported here so tests and BM25-only code do not load PyTorch.
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def embed_texts(model, texts: list[str]) -> list[list[float]]:
    """Embed passages. Vectors are normalized to length 1 so that cosine
    similarity is a plain dot product. Passages get no instruction prefix."""
    vectors = model.encode(texts, batch_size=EMBED_BATCH_SIZE, normalize_embeddings=True,
                           show_progress_bar=False)
    return [v.tolist() for v in vectors]


def embed_query(model, query: str) -> list[float]:
    """Embed a question, with the prefix the model card asks for on queries."""
    vector = model.encode([cfg.EMBEDDING_QUERY_PREFIX + query], normalize_embeddings=True,
                          show_progress_bar=False)[0]
    return vector.tolist()


def token_lengths(model, texts: list[str]) -> list[int]:
    """Model tokens per text, to see whether any chunk exceeds the model's
    input limit (anything past the limit is silently cut off when embedding)."""
    return [len(model.tokenizer(t, truncation=False)["input_ids"]) for t in texts]


def build_chroma(chunks: list[dict], embeddings: list[list[float]], chroma_dir: Path,
                 meta: dict, collection_name: str = cfg.CHROMA_COLLECTION):
    """Create the vector collection from scratch and return it."""
    import chromadb

    if len(chunks) != len(embeddings):
        raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
    client = chromadb.PersistentClient(path=str(chroma_dir))
    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(collection_name)
    collection = client.create_collection(
        collection_name,
        configuration={"hnsw": {"space": "cosine"}},   # distance = 1 - cosine similarity
        metadata=meta or None,
        embedding_function=None,                       # we pass our own embeddings
    )
    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[{k: c.get(k, "") for k in METADATA_FIELDS} for c in chunks],
    )
    return collection


def open_collection(chroma_dir: Path = cfg.CHROMA_DIR, collection_name: str = cfg.CHROMA_COLLECTION):
    import chromadb

    return chromadb.PersistentClient(path=str(chroma_dir)).get_collection(
        collection_name, embedding_function=None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=STRATEGIES, default=cfg.CHUNK_STRATEGY)
    ap.add_argument("--max-words", type=int, default=cfg.CHUNK_MAX_WORDS)
    a = ap.parse_args()

    chunks = chunk_docs(load_jsonl(cfg.DOCS_PATH), strategy=a.strategy, max_words=a.max_words)
    save_jsonl(cfg.CHUNKS_PATH, chunks)
    print(f"chunks: {len(chunks)} ({stats(chunks)})")
    meta = {"built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "strategy": a.strategy, "max_words": a.max_words, "n_chunks": len(chunks),
            "embedding_model": cfg.EMBEDDING_MODEL, "catalog_year": cfg.CATALOG_YEAR}

    Bm25Index.build(chunks).save(cfg.BM25_PATH, meta=meta)
    print(f"bm25: {len(chunks)} chunks -> {cfg.BM25_PATH}")

    model = load_embedder()
    texts = [c["text"] for c in chunks]
    lengths = token_lengths(model, texts)
    over = [c["id"] for c, n in zip(chunks, lengths) if n > model.max_seq_length]
    print(f"embedding with {cfg.EMBEDDING_MODEL}: longest chunk {max(lengths)} tokens, "
          f"limit {model.max_seq_length}, over limit: {over or 'none'}")
    collection = build_chroma(chunks, embed_texts(model, texts), cfg.CHROMA_DIR, meta)
    print(f"chroma: {collection.count()} vectors -> {cfg.CHROMA_DIR}")


if __name__ == "__main__":
    main()
