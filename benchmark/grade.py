"""Grade the blind sheet by hand, one response at a time.

  python -m benchmark.grade               # grade everything not graded yet
  python -m benchmark.grade --status      # how many are done
  python -m benchmark.grade --redo s014-B # grade one item again

Each screen shows the question, the expected answer, the source pages and one
response labelled A or B. It never shows which system wrote it and this tool
never opens the answer key. Read GRADING_RUBRIC.md first; the label is chosen
with a number key:

  1 correct   2 partially_correct   3 incorrect
  4 correct_refusal   5 wrong_refusal   6 hallucinated

Other keys: o opens the source pages in the browser, s skips for now, q quits.
After the label you can type a note (Enter for none). A note is required for
"hallucinated": quote the false claim. Every grade is saved at once, so you
can stop and continue later.
"""
from __future__ import annotations

import argparse
import csv
import sys
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from benchmark.make_blind_sheet import GRADING_DIR, SHEET_PATH

GRADES_PATH = GRADING_DIR / "human_grades.csv"
GRADE_COLUMNS = ("item", "id", "response_label", "label", "notes", "graded_at")
LABEL_KEYS = {"1": "correct", "2": "partially_correct", "3": "incorrect",
              "4": "correct_refusal", "5": "wrong_refusal", "6": "hallucinated"}
LEGEND = "  ".join(f"{key} {label}" for key, label in LABEL_KEYS.items())
PROMPT = "Type 1-6 to grade, o = open source pages, s = skip, q = quit: "
RULE = "=" * 78


def load_sheet(path: Path = SHEET_PATH) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_grades(path: Path = GRADES_PATH) -> dict[str, dict]:
    """Item -> its grade row. A later row for the same item replaces an
    earlier one, which is how --redo works without rewriting the file."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        return {row["item"]: row for row in csv.DictReader(f)}


def append_grade(path: Path, sheet_row: dict, label: str, notes: str) -> None:
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GRADE_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({"item": sheet_row["item"], "id": sheet_row["id"],
                         "response_label": sheet_row["response_label"], "label": label, "notes": notes,
                         "graded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})


def format_item(row: dict, position: int, total: int) -> str:
    sources = row["source_urls"].split(" | ") if row["source_urls"] else ["(none)"]
    return "\n".join([
        "", RULE, f"{position} of {total}   item {row['item']}", RULE,
        f"Question: {row['question']}",
        f"Expected answer: {row['expected_answer']}",
        "Source pages:", *(f"  {url}" for url in sources),
        "", f"Response {row['response_label']}:", row["response_text"], "", LEGEND, "",
    ])


def _ask(stream, out, prompt: str) -> str | None:
    """One typed line without its newline, or None when input has ended."""
    out.write(prompt)
    out.flush()
    line = stream.readline()
    return None if line == "" else line.strip()


def _ask_grade(row: dict, stream, out, opener: Callable[[str], object]) -> tuple[str, str] | str | None:
    """(label, notes), or "skip", or None to stop (q or end of input)."""
    while True:
        typed = _ask(stream, out, PROMPT)
        if typed is None or typed.lower() == "q":
            return None
        if typed.lower() == "s":
            return "skip"
        if typed.lower() == "o":
            for url in filter(None, row["source_urls"].split(" | ")):
                opener(url)
            continue
        if typed not in LABEL_KEYS:
            out.write(f"'{typed}' is not a choice.\n")
            continue
        label = LABEL_KEYS[typed]
        notes = _ask(stream, out, f"{label}. Notes (Enter for none): ")
        while notes == "" and label == "hallucinated":
            notes = _ask(stream, out, "Required for hallucinated. Quote the false claim: ")
        if notes is None:
            return None
        return label, notes


def grade_counts(path: Path = GRADES_PATH) -> dict[str, int]:
    """Item -> how many times it has been graded (2 or more means regraded)."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        counts: dict[str, int] = {}
        for row in csv.DictReader(f):
            counts[row["item"]] = counts.get(row["item"], 0) + 1
        return counts


def read_item_list(path: Path) -> list[str]:
    """Item ids from a text file, one per line; blank lines and # comments are ignored."""
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def run(sheet_path: Path = SHEET_PATH, grades_path: Path = GRADES_PATH, stream=sys.stdin, out=sys.stdout,
        redo: str | list[str] | None = None, opener: Callable[[str], object] = webbrowser.open,
        skip_regraded: bool = False) -> int:
    """Grade the pending items, or the items in `redo` again. With
    `skip_regraded`, redo items that already have a second grade are left
    out, so a long redo list can be stopped and continued. Returns how many
    grades were saved."""
    sheet = load_sheet(sheet_path)
    graded = load_grades(grades_path)
    wanted = [redo] if isinstance(redo, str) else list(redo or [])
    unknown = sorted(set(wanted) - {r["item"] for r in sheet})
    if unknown:
        raise ValueError(f"not on the sheet: {', '.join(unknown)}")
    if wanted:
        counts = grade_counts(grades_path) if skip_regraded else {}
        todo = [r for r in sheet if r["item"] in wanted and counts.get(r["item"], 0) < 2]
    else:
        todo = [r for r in sheet if r["item"] not in graded]
    saved = 0
    for position, row in enumerate(todo, start=1):
        out.write(format_item(row, position, len(todo)))
        answer = _ask_grade(row, stream, out, opener)
        if answer is None:
            break
        if answer == "skip":
            continue
        append_grade(grades_path, row, *answer)
        saved += 1
    out.write(f"\nsaved {saved} this session; {len(load_grades(grades_path))} of {len(sheet)} graded in total.\n")
    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description="Hand-grade the blind sheet.")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--redo", metavar="ITEM", nargs="+", help="grade these items again, e.g. s014-B")
    ap.add_argument("--redo-file", type=Path, metavar="FILE",
                    help="grade again every item listed in FILE (one per line); can be stopped and continued")
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stdin):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    if not SHEET_PATH.exists():
        raise SystemExit(f"{SHEET_PATH} not found; run `python -m benchmark.make_blind_sheet` first")
    if a.status:
        sheet, graded = load_sheet(), load_grades()
        left = [r["item"] for r in sheet if r["item"] not in graded]
        print(f"{len(sheet) - len(left)} of {len(sheet)} graded; {len(left)} left")
        if a.redo_file:
            counts, listed = grade_counts(), read_item_list(a.redo_file)
            done = sum(1 for item in listed if counts.get(item, 0) >= 2)
            print(f"{done} of {len(listed)} items in {a.redo_file.name} regraded; {len(listed) - done} left")
        return
    if a.redo_file:
        run(redo=read_item_list(a.redo_file), skip_regraded=True)
    else:
        run(redo=a.redo)


if __name__ == "__main__":
    main()
