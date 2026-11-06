"""Stratified random split of questions_all.jsonl into dev (~40%) and test (~60%).

  python -m benchmark.split --seed 20261030

Why stratified: every category gets the same dev/test proportion, so the dev
set is a faithful miniature of the test set and per-category test numbers
are not starved. Why a fixed seed: the split is reproducible and auditable.
The script refuses to overwrite an existing locked test set (see lock.py).
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict

from benchmark import lock
from benchmark.questions import (ALL_PATH, DEV_PATH, TEST_PATH, load, save,
                                 summarize, validate)

DEV_FRACTION = 0.4


def stratified_split(questions: list[dict], seed: int,
                     dev_fraction: float = DEV_FRACTION) -> tuple[list[dict], list[dict]]:
    """Return (dev, test). Pure function: same inputs, same output."""
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        by_cat[q["category"]].append(q)
    dev: list[dict] = []
    test: list[dict] = []
    for cat in sorted(by_cat):
        items = sorted(by_cat[cat], key=lambda q: q["id"])
        rng.shuffle(items)
        n_dev = round(len(items) * dev_fraction)
        dev.extend(items[:n_dev])
        test.extend(items[n_dev:])
    return sorted(dev, key=lambda q: q["id"]), sorted(test, key=lambda q: q["id"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    if lock.LOCK_PATH.exists():
        raise SystemExit(f"{lock.LOCK_PATH} exists: the test set is locked. "
                         "Delete the lock on purpose if you really mean to re-split.")
    questions = load(ALL_PATH)
    problems = validate(questions)
    if problems:
        raise SystemExit("question set invalid:\n  " + "\n  ".join(problems))
    dev, test = stratified_split(questions, a.seed)
    save(DEV_PATH, dev)
    save(TEST_PATH, test)
    print("dev :", summarize(dev))
    print("test:", summarize(test))
    print(f"seed {a.seed}; now run: python -m benchmark.lock")


if __name__ == "__main__":
    main()
