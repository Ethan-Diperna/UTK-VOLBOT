"""Find the catalog chunks most relevant to a question. Four selectable modes.

  python -m src.retrieve "what are the prereqs for cosc 360"
  python -m src.retrieve "classes about hacking" --mode vector --k 3
  python -m src.retrieve "math323 prereqs" --mode all          # compare modes

Modes:
  vector         embedding similarity only (meaning, weak on exact codes)
  bm25           keyword match only (exact codes, blind to synonyms)
  hybrid         both, merged with Reciprocal Rank Fusion
  hybrid_rerank  hybrid candidates re-scored by a cross-encoder

Run `python -m src.index` first; this module only reads the indexes.
"""
from __future__ import annotations

import argparse
import functools
import sys
from typing import Callable

from src import config as cfg
from src.chunk import load_jsonl
from src.index import Bm25Index, embed_query, load_embedder, open_collection

MODES = ("vector", "bm25", "hybrid", "hybrid_rerank")
Scored = list[tuple[str, float]]   # (chunk id, score), best first


def rrf(rankings: list[list[str]], k: int = cfg.RRF_K) -> Scored:
    """Reciprocal Rank Fusion: merge several ranked lists into one.

    Each list votes for a chunk with 1 / (k + rank), rank starting at 1, and
    the votes are added up:

        score(chunk) = sum over lists of 1 / (k + rank of chunk in that list)

    Why ranks and not the raw scores: a BM25 score (say 9.5) and a cosine
    similarity (say 0.74) are on unrelated scales, so adding them would let one
    search drown out the other. Ranks are comparable. A chunk both searches
    place near the top collects two large votes and beats a chunk only one
    search likes. k (60 by convention) damps the top of the list so rank 1 is
    not worth vastly more than rank 2. Ties keep first-appearance order."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


class Retriever:
    """Holds the indexes and runs a search in any mode.

    `vector_search(query, k)` returns (chunk id, similarity) pairs and
    `rerank(query, texts)` returns one relevance score per text. They are
    passed in as plain functions so tests can swap in fakes and never load a
    model; `from_disk` wires up the real ones."""

    def __init__(self, chunks: dict[str, dict], bm25: Bm25Index,
                 vector_search: Callable[[str, int], Scored],
                 rerank: Callable[[str, list[str]], list[float]]):
        self.chunks = chunks
        self.bm25 = bm25
        self.vector_search = vector_search
        self.rerank = rerank

    @classmethod
    def from_disk(cls) -> "Retriever":
        chunks = {c["id"]: c for c in load_jsonl(cfg.CHUNKS_PATH)}
        bm25 = Bm25Index.load(cfg.BM25_PATH)
        collection = open_collection()
        if set(bm25.ids) != set(chunks) or collection.count() != len(chunks):
            raise RuntimeError("indexes do not match chunks.jsonl; run `python -m src.index`")

        # Models load on first use: BM25-only searches never pay for PyTorch.
        @functools.lru_cache(maxsize=1)
        def embedder():
            return load_embedder()

        @functools.lru_cache(maxsize=1)
        def reranker():
            from sentence_transformers import CrossEncoder
            return CrossEncoder(cfg.RERANKER_MODEL)

        def vector_search(query: str, k: int) -> Scored:
            res = collection.query(query_embeddings=[embed_query(embedder(), query)], n_results=k)
            # Chroma returns cosine distance; 1 - distance is cosine similarity.
            return [(cid, 1.0 - dist) for cid, dist in zip(res["ids"][0], res["distances"][0])]

        def rerank(query: str, texts: list[str]) -> list[float]:
            # A cross-encoder reads the question and one chunk together and
            # outputs a relevance score. More accurate than comparing two
            # separately made embeddings, but too slow to run on every chunk,
            # so it only re-scores the short candidate list.
            pairs = [(query, text) for text in texts]
            return [float(s) for s in reranker().predict(pairs, show_progress_bar=False)]

        return cls(chunks, bm25, vector_search, rerank)

    def _fused(self, query: str, candidates: int) -> Scored:
        vector_ids = [cid for cid, _ in self.vector_search(query, candidates)]
        bm25_ids = [cid for cid, _ in self.bm25.search(query, candidates)]
        return rrf([vector_ids, bm25_ids])

    def search(self, query: str, mode: str = cfg.RETRIEVAL_MODE, k: int = cfg.TOP_K,
               candidates: int = cfg.RETRIEVAL_CANDIDATES) -> list[dict]:
        """Return the top-k chunks, each with `rank`, `score` and `mode` added.
        Scores are only comparable within one mode."""
        if mode not in MODES:
            raise ValueError(f"unknown retrieval mode {mode!r}; choose from {MODES}")
        query = query.strip()
        if not query:
            raise ValueError("query is empty")
        if mode == "vector":
            scored = self.vector_search(query, k)
        elif mode == "bm25":
            scored = self.bm25.search(query, k)
        elif mode == "hybrid":
            scored = self._fused(query, candidates)[:k]
        else:
            pool = [cid for cid, _ in self._fused(query, candidates)[:candidates]]
            scores = self.rerank(query, [self.chunks[cid]["text"] for cid in pool])
            scored = sorted(zip(pool, scores), key=lambda pair: pair[1], reverse=True)[:k]
        return [{**self.chunks[cid], "rank": rank, "score": float(score), "mode": mode}
                for rank, (cid, score) in enumerate(scored, start=1)]


def format_hits(hits: list[dict], preview_chars: int = 160) -> str:
    lines = []
    for h in hits:
        where = h["title"] + (f" / {h['section']}" if h["section"] else "")
        preview = " ".join(h["body"].split())[:preview_chars] if "body" in h else ""
        lines.append(f"{h['rank']:>2}. {h['score']:>8.4f}  {h['id']}  {where}\n      {preview}\n      {h['url']}")
    return "\n".join(lines) or "(no results)"


def main() -> None:
    ap = argparse.ArgumentParser(description="Show ranked catalog chunks for a question.")
    ap.add_argument("query")
    ap.add_argument("--mode", choices=MODES + ("all",), default=cfg.RETRIEVAL_MODE)
    ap.add_argument("--k", type=int, default=cfg.TOP_K)
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # catalog text has non-ASCII characters
    retriever = Retriever.from_disk()
    for mode in (MODES if a.mode == "all" else (a.mode,)):
        print(f"\n== {mode} ==")
        print(format_hits(retriever.search(a.query, mode=mode, k=a.k)))


if __name__ == "__main__":
    main()
