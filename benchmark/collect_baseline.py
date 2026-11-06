"""Record the baseline assistant's answers by hand, one question at a time.

  python -m benchmark.collect_baseline --split dev
  python -m benchmark.collect_baseline --split test
  python -m benchmark.collect_baseline --split dev --status

Flow per question: the exact question text is printed and copied to the
clipboard (Windows `clip`). Paste it into a FRESH UT Verse conversation, copy
the full answer, paste it here, then type a line containing only `.` to save.
Commands at the prompt: `.` save, `s` skip for now, `q` quit. Progress is
appended to benchmark/baseline_responses/<split>.jsonl, so you can stop and
resume. Already-answered ids are skipped.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmark.questions import BENCH_DIR, DEV_PATH, TEST_PATH, load
from src import config as cfg

RESPONSES_DIR = BENCH_DIR / "baseline_responses"
END_MARK = "."


def responses_path(split: str) -> Path:
    return RESPONSES_DIR / f"{split}.jsonl"


def load_responses(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {r["id"]: r for r in rows}


def append_response(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def copy_to_clipboard(text: str) -> bool:
    try:
        subprocess.run(["clip"], input=text.encode("utf-16le"), check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def read_pasted(stream=sys.stdin) -> tuple[str, str]:
    """Read lines until a line that is only END_MARK. Returns (command, text).
    command is 'save', 'skip', 'quit', or 'eof'."""
    lines: list[str] = []
    for raw in stream:
        line = raw.rstrip("\r\n")
        if line == END_MARK:
            return "save", "\n".join(lines).strip()
        if not lines and line.strip().lower() == "s":
            return "skip", ""
        if not lines and line.strip().lower() == "q":
            return "quit", ""
        lines.append(line)
    return "eof", "\n".join(lines).strip()


def make_row(q: dict, response: str, variant: str) -> dict:
    return {
        "id": q["id"],
        "question": q["question"],  # exact text as shown and copied
        "response": response,
        "system": cfg.BASELINE_NAME,
        "system_url": cfg.BASELINE_URL,
        "assistant_variant": variant,
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fresh_conversation": True,
        "notes": "",
    }


def show_status(questions: list[dict], done: dict[str, dict], out=sys.stdout) -> None:
    missing = [q["id"] for q in questions if q["id"] not in done]
    print(f"{len(done)}/{len(questions)} collected; remaining: {', '.join(missing) or 'none'}", file=out)


def run(split: str, variant: str, stream=sys.stdin, out=sys.stdout,
        questions_path: Path | None = None, out_path: Path | None = None) -> int:
    questions = load(questions_path or (DEV_PATH if split == "dev" else TEST_PATH))
    path = out_path or responses_path(split)
    done = load_responses(path)
    saved = 0
    for q in questions:
        if q["id"] in done:
            continue
        remaining = len(questions) - len(done)
        print(f"\n[{q['id']}] ({remaining} remaining)  category={q['category']}", file=out)
        print("=" * 72, file=out)
        print(q["question"], file=out)
        print("=" * 72, file=out)
        clip = copy_to_clipboard(q["question"])
        print(("(copied to clipboard) " if clip else "") +
              f"Paste the answer, then a line with only `{END_MARK}`. `s` skip, `q` quit.", file=out)
        cmd, text = read_pasted(stream)
        if cmd == "skip":
            continue
        if cmd in ("quit", "eof"):
            break
        if not text:
            print("empty response, not saved", file=out)
            continue
        row = make_row(q, text, variant)
        append_response(path, row)
        done[q["id"]] = row
        saved += 1
        print(f"saved {q['id']} ({len(text)} chars) -> {path.name}", file=out)
    show_status(questions, done, out)
    return saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--variant", default="",
                    help="which UT Verse gallery assistant you used, e.g. 'default chat'")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.status:
        questions = load(DEV_PATH if a.split == "dev" else TEST_PATH)
        show_status(questions, load_responses(responses_path(a.split)))
        return
    variant = a.variant or input("Which UT Verse assistant are you using (name as shown in the gallery)? ").strip()
    print(f"Baseline: {cfg.BASELINE_NAME} / {variant}. Use a fresh conversation for every question.")
    run(a.split, variant)


if __name__ == "__main__":
    main()
