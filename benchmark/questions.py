"""Question-set schema, loading, and validation.

File format: JSON Lines. One question per line:
  {"id": "q001",
   "question": "How many credit hours is COSC 302?",
   "expected_answer": "4",
   "source_urls": ["https://catalog.utk.edu/preview_course_nopop.php?catoid=56&coid=367193"],
   "category": "single_course",
   "origin": "self",
   "notes": ""}

Unanswerable questions have expected_answer describing the correct refusal
(e.g. "Not in the catalog; it does not list instructors") and source_urls [].
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
ALL_PATH = BENCH_DIR / "questions_all.jsonl"
DEV_PATH = BENCH_DIR / "questions_dev.jsonl"
TEST_PATH = BENCH_DIR / "questions_test.jsonl"

CATEGORIES = (
    "single_course",     # one fact about one course (credit hours, title, Vol Core)
    "prerequisite",      # prerequisites / corequisites of one course
    "program",           # degree / program / certificate requirements
    "multi_hop",         # needs two or more pages (prerequisite chains, course in program)
    "comparison",        # difference between two courses or programs
    "unanswerable",      # not covered by the catalog; correct behavior is to say so
)
ORIGINS = (
    "self",              # written by the project author
    "classmate",         # a question a friend or classmate actually asked
    "typo_slang",        # realistic informal phrasing, typos, abbreviations
    "catalog_derived",   # drafted by reading the catalog page (then hand-verified)
)
REQUIRED = ("id", "question", "expected_answer", "source_urls", "category", "origin")
CATALOG_URL_PREFIX = "https://catalog.utk.edu/"


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save(path: Path, questions: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")


def validate(questions: list[dict]) -> list[str]:
    """Return a list of problems. Empty list means valid."""
    problems: list[str] = []
    ids = Counter(q.get("id") for q in questions)
    for dup, n in ids.items():
        if n > 1:
            problems.append(f"duplicate id {dup} ({n}x)")
    for q in questions:
        qid = q.get("id", "<no id>")
        for key in REQUIRED:
            if key not in q:
                problems.append(f"{qid}: missing {key}")
        if q.get("category") not in CATEGORIES:
            problems.append(f"{qid}: bad category {q.get('category')!r}")
        if q.get("origin") not in ORIGINS:
            problems.append(f"{qid}: bad origin {q.get('origin')!r}")
        if not str(q.get("question", "")).strip() or not str(q.get("expected_answer", "")).strip():
            problems.append(f"{qid}: empty question or expected_answer")
        urls = q.get("source_urls", [])
        if not isinstance(urls, list):
            problems.append(f"{qid}: source_urls must be a list")
            continue
        if q.get("category") == "unanswerable":
            if urls:
                problems.append(f"{qid}: unanswerable questions must have no source_urls")
        elif not urls:
            problems.append(f"{qid}: answerable question needs at least one source url")
        for u in urls:
            if not str(u).startswith(CATALOG_URL_PREFIX):
                problems.append(f"{qid}: source url not on the catalog site: {u}")
    return problems


def summarize(questions: list[dict]) -> str:
    cats = Counter(q["category"] for q in questions)
    origins = Counter(q["origin"] for q in questions)
    return (f"{len(questions)} questions; by category {dict(cats)}; "
            f"by origin {dict(origins)}")


if __name__ == "__main__":
    import sys
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ALL_PATH
    qs = load(target)
    errs = validate(qs)
    print(summarize(qs))
    for e in errs:
        print("PROBLEM:", e)
    sys.exit(1 if errs else 0)
