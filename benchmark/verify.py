"""Hand-verify drafted questions one at a time.

  python -m benchmark.verify                       # all unverified rows in both draft files
  python -m benchmark.verify benchmark/questions_self.jsonl
  python -m benchmark.verify --status
  python -m benchmark.verify --no-open             # do not open source pages in the browser

Flow per row: the question, drafted answer, and source URLs are printed and the
source pages are opened in your default browser. Compare, then type one of:
  y   answer matches the page: mark verified
  e   answer is wrong or incomplete: type the corrected answer, end with a line
      containing only `.`, then it is marked verified
  d   drop this question from the file
  s   skip for now (stays unverified)
  q   quit
The file is rewritten after every decision, so you can stop and resume.
"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

from benchmark.questions import BENCH_DIR, load, save, validate

DRAFT_FILES = (
    BENCH_DIR / "questions_catalog_derived.jsonl",
    BENCH_DIR / "questions_self.jsonl",
)
END_MARK = "."
COMMANDS = {"y", "e", "d", "s", "q"}


def is_verified(row: dict) -> bool:
    return bool(row.get("verified_by_hand"))


def apply_decision(rows: list[dict], qid: str, cmd: str, new_answer: str = "",
                   now: str | None = None) -> list[dict]:
    """Return a new list with the decision applied to the row with id `qid`.
    cmd: 'y' verify as-is, 'e' replace answer and verify, 'd' drop the row.
    Other commands return the rows unchanged. Never mutates the input."""
    stamp = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    out: list[dict] = []
    for r in rows:
        if r["id"] != qid:
            out.append(r)
            continue
        if cmd == "d":
            continue
        if cmd == "y":
            out.append({**r, "verified_by_hand": True, "verified_at": stamp})
        elif cmd == "e":
            if not new_answer.strip():
                raise ValueError("corrected answer must not be empty")
            out.append({**r, "expected_answer": new_answer.strip(),
                        "verified_by_hand": True, "verified_at": stamp})
        else:
            out.append(r)
    return out


def read_command(stream=sys.stdin) -> str:
    for raw in stream:
        cmd = raw.strip().lower()
        if cmd in COMMANDS:
            return cmd
        print(f"type one of {' '.join(sorted(COMMANDS))}", file=sys.stderr)
    return "q"


def read_answer(stream=sys.stdin) -> str:
    lines: list[str] = []
    for raw in stream:
        line = raw.rstrip("\r\n")
        if line == END_MARK:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def show(row: dict, remaining: int, out) -> None:
    print(f"\n[{row['id']}] ({remaining} left)  category={row['category']}  origin={row['origin']}", file=out)
    print("=" * 72, file=out)
    print("Q:", row["question"], file=out)
    print("-" * 72, file=out)
    print("A:", row["expected_answer"], file=out)
    if row.get("notes"):
        print("notes:", row["notes"], file=out)
    for u in row["source_urls"]:
        print("src:", u, file=out)
    print("=" * 72, file=out)
    print("y verified / e edit answer / d drop / s skip / q quit >", end=" ", file=out, flush=True)


def status(path: Path, out=sys.stdout) -> None:
    rows = load(path)
    done = sum(is_verified(r) for r in rows)
    print(f"{path.name}: {done}/{len(rows)} verified", file=out)


def run(path: Path, stream=sys.stdin, out=sys.stdout, open_browser: bool = True) -> int:
    """Walk unverified rows in `path`. Returns number of decisions saved."""
    rows = load(path)
    saved = 0
    for row in list(rows):
        if is_verified(row):
            continue
        remaining = sum(not is_verified(r) for r in rows)
        show(row, remaining, out)
        if open_browser:
            for u in row["source_urls"]:
                webbrowser.open(u)
        cmd = read_command(stream)
        if cmd == "q":
            break
        if cmd == "s":
            continue
        new_answer = ""
        if cmd == "e":
            print(f"Type the corrected answer, then a line with only `{END_MARK}`:", file=out)
            new_answer = read_answer(stream)
            if not new_answer:
                print("empty answer, nothing changed", file=out)
                continue
        rows = apply_decision(rows, row["id"], cmd, new_answer)
        problems = validate(rows)
        if problems:
            print("NOT SAVED, validation failed:", *problems, sep="\n  ", file=out)
            rows = load(path)
            continue
        save(path, rows)
        saved += 1
        print({"y": "verified", "e": "edited + verified", "d": "dropped"}[cmd], row["id"], file=out)
    status(path, out)
    return saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", type=Path, help="question JSONL files (default: both draft files)")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--no-open", action="store_true", help="do not open source pages in the browser")
    a = ap.parse_args()
    files = a.files or [p for p in DRAFT_FILES if p.exists()]
    for path in files:
        if a.status:
            status(path)
        else:
            run(path, open_browser=not a.no_open)


if __name__ == "__main__":
    main()
