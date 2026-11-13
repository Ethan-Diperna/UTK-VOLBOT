"""Build the blind grading sheet for the head-to-head on the test set.

  python -m benchmark.make_blind_sheet

For every test question the two systems' answers are put side by side as
"Response A" and "Response B", in an order decided by a seeded coin flip. The
sheet (grading/blind_sheet.csv) never names a system. The answer key
(grading/blind_key.json) is written separately and must stay unopened until
grading is finished; score.py is the only thing that reads it.

Both answers go through the same clean_response(), which removes what would
give a system away at a glance: citation markers, the "Sources" trailer of the
baseline's copy output, markdown emphasis, curly quotes. Wording is never
changed, so a phrase such as "the excerpts" can still hint at the author. That
limit is stated in the README. Citations are graded in a separate pass,
because they cannot be shown without revealing the system.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

from benchmark import freeze, lock
from benchmark.collect_baseline import load_responses, responses_path
from benchmark.questions import BENCH_DIR, TEST_PATH, load
from src import config as cfg
from src.chunk import load_jsonl

GRADING_DIR = BENCH_DIR / "grading"
SHEET_PATH = GRADING_DIR / "blind_sheet.csv"
KEY_PATH = GRADING_DIR / "blind_key.json"
BLIND_SEED = 20261113
SHEET_COLUMNS = ("item", "id", "question", "expected_answer", "source_urls",
                 "response_label", "response_text")

# A line that holds nothing but a citation: "3" (copy-button output) or
# "[3](https://...)" (rich-text paste).
MARKER_LINE_RE = re.compile(r"^\s*(?:\d+|\[\d+\]\(https?://[^)\s]*\))\s*$")
LINK_CITATION_RE = re.compile(r"\s*\[\d+\]\(https?://[^)\s]*\)")
# "[1]", "[2][3]", "[1, 4]" and ranges like "[1]-[11]", with the space before them.
CITATION_RE = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\](?:\s*[–-]\s*\[\d+\])?")
# Copied baseline answers end with the page's sources bar: "Sources / 9 / 4 Referenced /
# Learn More / 6 Images", sometimes with "Click to view sources" in it.
TRAILER_RE = re.compile(r"\n\s*Sources\s*(?:\n\s*(?:\d+|Click to view sources|\d+ Referenced|"
                        r"Learn More|\d+ Images?)\s*)*\s*$")
QUOTES = {"‘": "'", "’": "'", "“": '"', "”": '"', " ": " "}


def _drop_marker_lines(text: str) -> str:
    """Remove citation-only lines. The copy button breaks a sentence around
    them ("...Algebra I.\\n1\\n Some older...") and marks the continuation with
    a leading space, so such a line is joined back onto the one before."""
    out: list[str] = []
    after_marker = False
    for line in text.split("\n"):
        if MARKER_LINE_RE.match(line):
            after_marker = True
            continue
        if after_marker and line[:1] in (" ", "\t") and out and out[-1].strip():
            out[-1] = out[-1].rstrip() + " " + line.strip()
        else:
            out.append(line)
        after_marker = False
    return "\n".join(out)


def clean_response(text: str) -> str:
    """The answer as the grader sees it, with system-revealing formatting removed."""
    original = text.strip()
    text = text.replace("\r\n", "\n")
    for curly, plain in QUOTES.items():
        text = text.replace(curly, plain)
    text = TRAILER_RE.sub("", text)
    text = _drop_marker_lines(text)
    text = LINK_CITATION_RE.sub("", text)
    text = CITATION_RE.sub("", text)
    text = text.replace("**", "").replace("`", "")
    lines = [re.sub(r"^#+\s*", "", line) for line in text.split("\n")]
    lines = [re.sub(r"[ \t]{2,}", " ", line).rstrip() for line in lines]
    text = re.sub(r"[ \t]+([.,;:)])", r"\1", "\n".join(lines))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text or original   # an answer that was only a number must not vanish


def build_sheet(questions: list[dict], ours: dict[str, str], baseline: dict[str, str],
                seed: int = BLIND_SEED) -> tuple[list[dict], dict[str, dict[str, str]]]:
    """Return (sheet rows, key). key[question id] maps "A" and "B" to "ours"
    or "baseline". Raises if either system lacks an answer for a question."""
    missing = [f"{q['id']} ({name})" for q in questions
               for name, answers in (("ours", ours), ("baseline", baseline)) if not answers.get(q["id"])]
    if missing:
        raise ValueError(f"no answer for: {', '.join(missing)}")
    rng = random.Random(seed)
    rows: list[dict] = []
    key: dict[str, dict[str, str]] = {}
    for q in questions:
        first, second = ("ours", "baseline") if rng.random() < 0.5 else ("baseline", "ours")
        key[q["id"]] = {"A": first, "B": second}
        texts = {"ours": ours[q["id"]], "baseline": baseline[q["id"]]}
        for label in ("A", "B"):
            rows.append({
                "item": f"{q['id']}-{label}", "id": q["id"], "question": q["question"],
                "expected_answer": q["expected_answer"], "source_urls": " | ".join(q["source_urls"]),
                "response_label": label, "response_text": clean_response(texts[key[q["id"]][label]]),
            })
    return rows, key


def write_sheet(rows: list[dict], key: dict, sheet_path: Path = SHEET_PATH, key_path: Path = KEY_PATH,
                meta: dict | None = None) -> None:
    """Never overwrites: rebuilding a sheet after grading started would mix
    grades from two different A/B assignments."""
    if sheet_path.exists() or key_path.exists():
        raise FileExistsError(f"{sheet_path.name} or {key_path.name} already exists; not overwriting.")
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    with sheet_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    key_path.write_text(json.dumps({**(meta or {}), "assignments": key}, indent=2) + "\n",
                        encoding="utf-8", newline="\n")


def latest_test_run(results_dir: Path = cfg.RESULTS_DIR) -> Path:
    runs = sorted(p for p in results_dir.glob("*_test") if (p / "rows.jsonl").exists())
    if not runs:
        raise SystemExit("no test run found; run `python -m benchmark.run_eval --split test --answers` first")
    return runs[-1]


def our_answers(run_dir: Path) -> dict[str, str]:
    """Question id -> our system's answer text. A question whose generation
    failed is left out, which makes build_sheet refuse."""
    return {r["id"]: r["answer"] for r in load_jsonl(run_dir / "rows.jsonl") if r.get("answer")}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the blind A/B grading sheet for the test set.")
    ap.add_argument("--run", type=Path, help="test run folder (default: the newest *_test run)")
    a = ap.parse_args()
    if not lock.verify():
        raise SystemExit("questions_test.jsonl does not match test_set.sha256; refusing.")
    changed = freeze.changed_keys()
    if changed:
        raise SystemExit(f"config differs from the freeze in: {', '.join(changed)}; refusing.")
    run_dir = a.run or latest_test_run()
    questions = load(TEST_PATH)
    baseline = {qid: row["response"] for qid, row in load_responses(responses_path("test")).items()}
    rows, key = build_sheet(questions, our_answers(run_dir), baseline)
    meta = {"seed": BLIND_SEED, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_dir": run_dir.name, "n_questions": len(questions)}
    write_sheet(rows, key, meta=meta)
    print(f"wrote {len(rows)} responses for {len(questions)} questions -> {SHEET_PATH}")
    print(f"answer key -> {KEY_PATH}  (do not open until grading is finished)")


if __name__ == "__main__":
    main()
