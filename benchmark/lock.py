"""Lock the held-out test set with a SHA-256 hash.

  python -m benchmark.lock            # write test_set.sha256 (refuses if present)
  python -m benchmark.lock --verify   # exit 1 if questions_test.jsonl changed

The final evaluation (run_eval.py --split test) calls verify() first and
refuses to run if the hash does not match. Commit test_set.sha256 alongside
questions_test.jsonl so the git history shows when the set was frozen.
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from benchmark.questions import BENCH_DIR, TEST_PATH, load

LOCK_PATH = BENCH_DIR / "test_set.sha256"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_lock(path: Path = LOCK_PATH) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            k, _, v = line.partition(" ")
            fields[k] = v.strip()
    return fields


def write_lock(test_path: Path = TEST_PATH, lock_path: Path = LOCK_PATH) -> str:
    digest = file_sha256(test_path)
    lines = [f"sha256 {digest}",
             f"file {test_path.name}",
             f"count {len(load(test_path))}",
             f"locked_at {datetime.now(timezone.utc).isoformat(timespec='seconds')}"]
    lock_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return digest


def verify(test_path: Path = TEST_PATH, lock_path: Path = LOCK_PATH) -> bool:
    if not lock_path.exists():
        return False
    return read_lock(lock_path).get("sha256") == file_sha256(test_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        ok = verify()
        print("test set hash", "OK" if ok else "MISMATCH or no lock")
        raise SystemExit(0 if ok else 1)
    if LOCK_PATH.exists():
        raise SystemExit(f"{LOCK_PATH} already exists; refusing to overwrite.")
    print("locked:", write_lock())


if __name__ == "__main__":
    main()
