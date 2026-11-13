"""Turn ranked search hits into the excerpts the model actually reads.

Search returns the 5 best chunks. Dev-set error analysis (Phase 6) showed the
model almost never answered wrongly from them; it failed because a needed
piece was not among them. Three repairs, each a switchable step so the
ablation can measure it on its own:

  named_courses  A course code written in the question ("MATH 251") is looked
                 up directly, so its catalog entry is always present. Search
                 ranks by similarity and can crowd out one of two named courses.
  program_pages  A program page is indexed as ~15 small sections. A question
                 about the whole program ("which math classes do CS majors
                 take") needs many of them, so a hit on one section brings in
                 the whole page (small-to-big retrieval).
  prerequisites  For a course named in the question, follow its
                 "Prerequisite(s)" lines to those courses, and theirs, a few
                 hops. One search cannot find a chain: the question names only
                 the first link.

Nothing here looks at question wording beyond course codes, and nothing is
specific to a benchmark question.
"""
from __future__ import annotations

import re

from src import config as cfg

STEPS = ("named_courses", "program_pages", "prerequisites")
# 2-5 letters, optional space or hyphen, 3 digits: "COSC 360", "cosc360", "cosc-360".
CODE_RE = re.compile(r"\b([A-Za-z]{2,5})[\s-]?(\d{3})\b")
REQUISITE_MARK = "requisite"   # matches "Prerequisite(s):" and "Corequisite(s):" lines


def course_codes(text: str) -> list[str]:
    """Course codes in `text`, written the catalog's way ("COSC 360"), each
    once, in order of first appearance."""
    codes: list[str] = []
    for dept, number in CODE_RE.findall(text):
        code = f"{dept.upper()} {number}"
        if code not in codes:
            codes.append(code)
    return codes


def _tag(chunks: list[dict], step: str) -> list[dict]:
    return [{**c, "added_by": step} for c in chunks]


def _requisite_codes(chunk: dict) -> list[str]:
    lines = [line for line in chunk["text"].splitlines() if REQUISITE_MARK in line.lower()]
    return course_codes("\n".join(lines))


def _prerequisite_chunks(seeds: list[dict], by_code: dict[str, dict],
                         max_hops: int, max_added: int) -> list[dict]:
    """Breadth-first walk along prerequisite lines, nearest courses first."""
    seen = {c["id"] for c in seeds}
    added: list[dict] = []
    frontier = seeds
    for _ in range(max_hops):
        found: list[dict] = []
        for chunk in frontier:
            for code in _requisite_codes(chunk):
                target = by_code.get(code)
                if target is None or target["id"] in seen:
                    continue
                if len(added) + len(found) == max_added:
                    return added + found
                seen.add(target["id"])
                found.append(target)
        added = added + found
        frontier = found
    return added


def _with_program_pages(hits: list[dict], chunks: dict[str, dict], max_pages: int) -> list[dict]:
    """Hits in rank order, with each of the best `max_pages` program pages
    replaced by all of its sections in page order."""
    pages: list[str] = []
    for h in hits:
        if h["doc_type"] == "program" and h["doc_id"] not in pages and len(pages) < max_pages:
            pages.append(h["doc_id"])
    hit_by_id = {h["id"]: h for h in hits}
    out: list[dict] = []
    emitted: set[str] = set()
    for h in hits:
        if h["doc_id"] not in pages:
            out = out + _tag([h], "search")
        elif h["doc_id"] not in emitted:
            emitted.add(h["doc_id"])
            page = sorted((c for c in chunks.values() if c["doc_id"] == h["doc_id"]), key=lambda c: c["id"])
            out = out + [{**hit_by_id.get(c["id"], c), "added_by": "search" if c["id"] in hit_by_id else "program_pages"}
                         for c in page]
    return out


def build_context(question: str, hits: list[dict], chunks: dict[str, dict],
                  steps: tuple[str, ...] = cfg.CONTEXT_STEPS,
                  max_pages: int = cfg.CONTEXT_MAX_PROGRAM_PAGES,
                  max_hops: int = cfg.CONTEXT_PREREQ_MAX_HOPS,
                  max_added: int = cfg.CONTEXT_PREREQ_MAX_CHUNKS) -> list[dict]:
    """The chunks to send to the model, each tagged with `added_by` (the step
    that brought it in, or "search"). `chunks` is every chunk by id. With no
    steps the hits come back unchanged."""
    unknown = set(steps) - set(STEPS)
    if unknown:
        raise ValueError(f"unknown context steps {sorted(unknown)}; choose from {STEPS}")
    if not steps:
        return hits
    by_code = {c["code"]: c for c in chunks.values() if c["doc_type"] == "course" and c["code"]}
    named = [by_code[code] for code in course_codes(question) if code in by_code]
    hit_ids = {h["id"] for h in hits}

    parts: list[dict] = []
    if "named_courses" in steps:
        parts = parts + _tag([c for c in named if c["id"] not in hit_ids], "named_courses")
    parts = parts + (_with_program_pages(hits, chunks, max_pages) if "program_pages" in steps
                     else _tag(hits, "search"))
    if "prerequisites" in steps:
        parts = parts + _tag(_prerequisite_chunks(named, by_code, max_hops, max_added), "prerequisites")

    seen: set[str] = set()
    context: list[dict] = []
    for c in parts:
        if c["id"] not in seen:
            seen.add(c["id"])
            context.append(c)
    return context
