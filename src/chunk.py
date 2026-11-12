"""Split cleaned catalog docs into retrieval chunks.

  python -m src.chunk                                  # settings from src/config.py
  python -m src.chunk --strategy packed --max-words 300

Design, in the order the code runs:

1. A course page is one chunk. It is short (median ~50 words) and every fact
   about the course belongs together, so splitting it could only hurt.
2. A program page is long (up to ~2000 words) and the embedding model reads at
   most 512 tokens, so it must be split. We split where the catalog itself
   splits: at headings. A section stays whole if it fits in `max_words`;
   otherwise it is split at its sub-headings, and only as a last resort by
   lines, repeating `overlap_lines` lines so a requirement list cut in two
   still shares context.
3. Every program chunk starts with a header naming the program and section.
   "Term 3" means nothing alone; "Computer Science Major, BS / Term 3" can be
   retrieved and cited.
4. Strategy "section" emits those pieces as they are. Strategy "packed" merges
   neighbouring pieces up to `max_words` and repeats `overlap_sections` pieces
   between chunks: fewer, larger chunks. Which one retrieves better is decided
   on the dev set in Phase 6.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from src import config as cfg

HEADING_RE = re.compile(r"^(#{2,6})(?:\s+(.*))?$")
STRATEGIES = ("section", "packed")
OVERVIEW = "Overview"
# A section with fewer words than this (e.g. "Campus: Knoxville") is too small
# to be found on its own, so it is merged into the section that follows it.
MIN_SECTION_WORDS = 8


def n_words(text: str) -> int:
    return len(text.split())


def _words(lines: list[str]) -> int:
    return sum(n_words(line) for line in lines)


def split_blocks(text: str) -> list[dict]:
    """Parse cleaned page text into blocks: [{level, heading, path, lines}].

    The first block ("Overview", level 1) holds everything before the first
    heading. `path` is the chain of headings above and including the block.
    A heading with no text is a formatting artifact of the catalog's nested
    lists, so its lines continue the previous block."""
    blocks = [{"level": 1, "heading": OVERVIEW, "path": [OVERVIEW], "lines": []}]
    stack: list[tuple[int, str]] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        m = HEADING_RE.match(line)
        if not m:
            blocks[-1]["lines"].append(line)
            continue
        heading = (m.group(2) or "").strip()
        if not heading:
            continue
        level = len(m.group(1))
        stack = [(lv, h) for lv, h in stack if lv < level] + [(level, heading)]
        blocks.append({"level": level, "heading": heading,
                       "path": [h for _, h in stack], "lines": []})
    return blocks


def _render(group: list[dict]) -> list[str]:
    """Lines of a block followed by its sub-blocks, sub-headings kept as lines."""
    lines = list(group[0]["lines"])
    for b in group[1:]:
        lines = lines + [b["heading"]] + b["lines"]
    return lines


def _window_lines(lines: list[str], max_words: int, overlap_lines: int) -> list[list[str]]:
    """Greedy windows of whole lines, each at most `max_words` words (a single
    longer line is kept whole). Consecutive windows share `overlap_lines` lines."""
    windows: list[list[str]] = []
    current: list[str] = []
    fresh = 0  # lines in `current` that are not overlap from the previous window
    for line in lines:
        if current and fresh and _words(current) + n_words(line) > max_words:
            windows.append(current)
            tail = current[-overlap_lines:] if overlap_lines else []
            current = tail if _words(tail) + n_words(line) <= max_words else []
            fresh = 0
        current = current + [line]
        fresh += 1
    if fresh:
        windows.append(current)
    return windows


def _group_by_level(blocks: list[dict]) -> list[list[dict]]:
    """Cut a run of blocks into groups, each starting at the shallowest level."""
    if not blocks:
        return []
    top = min(b["level"] for b in blocks)
    groups: list[list[dict]] = []
    for b in blocks:
        if b["level"] == top or not groups:
            groups.append([b])
        else:
            groups[-1] = groups[-1] + [b]
    return groups


def _split_group(group: list[dict], max_words: int, overlap_lines: int) -> list[dict]:
    """Return pieces [{section, lines}] for one section and its sub-sections."""
    section = " > ".join(group[0]["path"])
    lines = _render(group)
    if _words(lines) <= max_words:
        return [{"section": section, "lines": lines}]
    if len(group) == 1:
        return [{"section": section, "lines": w}
                for w in _window_lines(lines, max_words, overlap_lines)]
    head = {**group[0]}
    pieces = _split_group([head], max_words, overlap_lines) if head["lines"] else []
    for sub in _group_by_level(group[1:]):
        pieces = pieces + _split_group(sub, max_words, overlap_lines)
    return pieces


def _merge_tiny(pieces: list[dict]) -> list[dict]:
    """Fold pieces under MIN_SECTION_WORDS into the next piece (or the previous
    one if they come last), keeping their heading as a line so nothing is lost."""
    merged: list[dict] = []
    carry: list[str] = []
    for p in pieces:
        label = p["section"].split(" > ")[-1]
        if _words(p["lines"]) < MIN_SECTION_WORDS:
            carry = carry + [label] + p["lines"]
            continue
        merged.append({"section": p["section"], "lines": carry + p["lines"]})
        carry = []
    if carry:
        if merged:
            merged[-1] = {**merged[-1], "lines": merged[-1]["lines"] + carry}
        else:
            merged.append({"section": OVERVIEW, "lines": carry})
    return merged


def _pack(pieces: list[dict], max_words: int, overlap_sections: int) -> list[dict]:
    """Merge neighbouring pieces up to `max_words`. Each new chunk starts with
    the last `overlap_sections` pieces of the previous chunk when they fit."""
    def size(ps: list[dict]) -> int:
        return sum(_words(p["lines"]) + n_words(p["section"]) for p in ps)

    def join(ps: list[dict]) -> dict:
        lines = list(ps[0]["lines"])
        for p in ps[1:]:
            lines = lines + [p["section"].split(" > ")[-1]] + p["lines"]
        names = [p["section"] for p in ps]
        section = names[0] if len(names) == 1 else f"{names[0]} to {names[-1]}"
        return {"section": section, "lines": lines}

    packed: list[dict] = []
    current: list[dict] = []
    fresh = 0
    for p in pieces:
        if current and fresh and size(current + [p]) > max_words:
            packed.append(join(current))
            tail = current[-overlap_sections:] if overlap_sections else []
            current = tail if size(tail + [p]) <= max_words else []
            fresh = 0
        current = current + [p]
        fresh += 1
    if fresh:
        packed.append(join(current))
    return packed


def _base(doc: dict) -> dict:
    meta = doc.get("metadata", {})
    return {"doc_id": doc["id"], "url": doc["url"], "title": doc["title"],
            "doc_type": doc["doc_type"], "dept": doc.get("dept") or "",
            "code": meta.get("code") or "", "catalog_year": meta.get("catalog_year") or ""}


def chunk_course(doc: dict) -> list[dict]:
    text = doc["text"].strip()
    return [{"id": doc["id"], **_base(doc), "section": "",
             "body": text, "text": text, "n_words": n_words(text)}]


def chunk_program(doc: dict, strategy: str, max_words: int,
                  overlap_lines: int, overlap_sections: int) -> list[dict]:
    blocks = split_blocks(doc["text"])
    overview = blocks[0]
    if overview["lines"] and overview["lines"][0] == doc["title"]:
        overview = {**overview, "lines": overview["lines"][1:]}  # title goes in the header
    pieces: list[dict] = []
    for group in [[overview]] + _group_by_level(blocks[1:]):
        if group[0]["lines"] or len(group) > 1 or group[0]["heading"] != OVERVIEW:
            pieces = pieces + _split_group(group, max_words, overlap_lines)
    pieces = _merge_tiny(pieces)
    if strategy == "packed":
        pieces = _pack(pieces, max_words, overlap_sections)
    chunks = []
    for i, p in enumerate(pieces):
        body = "\n".join(p["lines"])
        text = f"{doc['title']}\nSection: {p['section']}\n{body}"
        chunks.append({"id": f"{doc['id']}#{i:02d}", **_base(doc), "section": p["section"],
                       "body": body, "text": text, "n_words": n_words(text)})
    return chunks


def chunk_docs(docs: list[dict], strategy: str = cfg.CHUNK_STRATEGY,
               max_words: int = cfg.CHUNK_MAX_WORDS,
               overlap_lines: int = cfg.CHUNK_OVERLAP_LINES,
               overlap_sections: int = cfg.CHUNK_OVERLAP_SECTIONS) -> list[dict]:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown chunk strategy {strategy!r}; choose from {STRATEGIES}")
    chunks: list[dict] = []
    for doc in docs:
        if doc["doc_type"] == "course":
            chunks = chunks + chunk_course(doc)
        else:
            chunks = chunks + chunk_program(doc, strategy, max_words, overlap_lines, overlap_sections)
    return chunks


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stats(chunks: list[dict]) -> str:
    out = []
    for kind in ("course", "program"):
        sizes = [c["n_words"] for c in chunks if c["doc_type"] == kind]
        if sizes:
            out.append(f"{kind}: {len(sizes)} chunks, words min {min(sizes)} "
                       f"median {int(statistics.median(sizes))} max {max(sizes)}")
    return "; ".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=STRATEGIES, default=cfg.CHUNK_STRATEGY)
    ap.add_argument("--max-words", type=int, default=cfg.CHUNK_MAX_WORDS)
    ap.add_argument("--out", type=Path, default=cfg.CHUNKS_PATH)
    a = ap.parse_args()
    chunks = chunk_docs(load_jsonl(cfg.DOCS_PATH), strategy=a.strategy, max_words=a.max_words)
    save_jsonl(a.out, chunks)
    print(f"strategy={a.strategy} max_words={a.max_words}: wrote {len(chunks)} chunks -> {a.out}")
    print(stats(chunks))


if __name__ == "__main__":
    main()
