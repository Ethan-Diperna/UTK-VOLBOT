"""Have the LLM judge grade the same blind sheet the human graded.

  python -m benchmark.judge_sheet

The judge gets exactly what the human saw: the question, the expected answer,
the source pages and one cleaned response. It does not get the answer key, so
it is as blind as the human was. Its grades go to grading/judge_grades.csv and
are compared with the human grades by score.py. Items already judged are
skipped, so a run that stops half way (rate limit, no credits) can be rerun.
This costs API money: about one cent per response.
"""
from __future__ import annotations

import csv
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from benchmark.grade import load_grades, load_sheet
from benchmark.judge import JudgeError, judge
from benchmark.make_blind_sheet import GRADING_DIR, SHEET_PATH
from benchmark.questions import TEST_PATH, load
from src import config as cfg
from src.chunk import load_jsonl

JUDGE_GRADES_PATH = GRADING_DIR / "judge_grades.csv"
JUDGE_COLUMNS = ("item", "id", "response_label", "label", "notes", "judge_model", "graded_at")


def judge_items(sheet: list[dict], questions: list[dict], judge_fn, done: set[str],
                workers: int = cfg.EVAL_WORKERS) -> tuple[list[dict], list[tuple[str, str]]]:
    """Grade every sheet row whose item is not in `done`. Returns (grade rows
    in sheet order, [(item, error message)]). A failed item gets no row."""
    by_id = {q["id"]: q for q in questions}
    todo = [row for row in sheet if row["item"] not in done]

    def grade_one(row: dict) -> dict | str:
        try:
            result = judge_fn(by_id[row["id"]], row["response_text"])
        except (JudgeError, anthropic.APIError) as e:
            return str(e)
        return {"item": row["item"], "id": row["id"], "response_label": row["response_label"],
                "label": result["label"], "notes": result.get("notes", ""),
                "judge_model": result.get("judge_model", ""),
                "graded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(grade_one, todo))
    rows = [r for r in results if isinstance(r, dict)]
    errors = [(row["item"], r) for row, r in zip(todo, results) if isinstance(r, str)]
    return rows, errors


def append_rows(path: Path, rows: list[dict]) -> None:
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=JUDGE_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sheet = load_sheet(SHEET_PATH)
    titles = {d["url"]: d["title"] for d in load_jsonl(cfg.DOCS_PATH)}
    done = set(load_grades(JUDGE_GRADES_PATH))
    rows, errors = judge_items(sheet, load(TEST_PATH), lambda q, text: judge(q, text, page_titles=titles), done)
    append_rows(JUDGE_GRADES_PATH, rows)
    total = len(load_grades(JUDGE_GRADES_PATH))
    print(f"judged {len(rows)} now; {total} of {len(sheet)} in total; {len(errors)} failed")
    for item, message in errors[:5]:
        print(f"  {item}: {message[:160]}")


if __name__ == "__main__":
    main()
