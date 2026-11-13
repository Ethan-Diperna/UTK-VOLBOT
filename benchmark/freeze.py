"""Freeze the system's configuration before the test set is touched.

  python -m benchmark.freeze            # write frozen_config.json (refuses if present)
  python -m benchmark.freeze --verify   # exit 1 if any frozen setting has changed

Tuning happens on the dev set only. Once it is finished, this records every
setting that can change an answer: chunking, retrieval, context steps, models,
and hashes of the system prompt, the grading rubric and the chunk file.
`run_eval --split test` refuses to run unless the live config still matches,
so the test numbers provably come from the config chosen on dev. Commit the
file: the git history then shows it was frozen before the test run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from benchmark.judge import rubric_sha256
from benchmark.questions import BENCH_DIR
from src import config as cfg
from src.generate import SYSTEM_PROMPT

FROZEN_PATH = BENCH_DIR / "frozen_config.json"


def current_config() -> dict:
    """Every setting that can change what the system answers or how it is graded."""
    return {
        "catalog_year": cfg.CATALOG_YEAR,
        "chunk_strategy": cfg.CHUNK_STRATEGY,
        "chunk_max_words": cfg.CHUNK_MAX_WORDS,
        "chunk_overlap_lines": cfg.CHUNK_OVERLAP_LINES,
        "chunks_sha256": hashlib.sha256(cfg.CHUNKS_PATH.read_bytes()).hexdigest(),
        "embedding_model": cfg.EMBEDDING_MODEL,
        "reranker_model": cfg.RERANKER_MODEL,
        "retrieval_mode": cfg.RETRIEVAL_MODE,
        "top_k": cfg.TOP_K,
        "retrieval_candidates": cfg.RETRIEVAL_CANDIDATES,
        "rrf_k": cfg.RRF_K,
        "context_steps": list(cfg.CONTEXT_STEPS),
        "context_max_program_pages": cfg.CONTEXT_MAX_PROGRAM_PAGES,
        "context_prereq_max_hops": cfg.CONTEXT_PREREQ_MAX_HOPS,
        "context_prereq_max_chunks": cfg.CONTEXT_PREREQ_MAX_CHUNKS,
        "generation_model": cfg.GENERATION_MODEL,
        "generation_effort": cfg.GENERATION_EFFORT,
        "generation_max_tokens": cfg.GENERATION_MAX_TOKENS,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "judge_model": cfg.JUDGE_MODEL,
        "rubric_sha256": rubric_sha256(),
    }


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=cfg.ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_freeze(path: Path = FROZEN_PATH) -> dict:
    """Write the freeze file. Never overwrites: a second freeze would mean the
    config was changed after the first one."""
    if path.exists():
        raise FileExistsError(f"{path} already exists; the config is already frozen.")
    frozen = {"frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "git_commit": _git_commit(),   # the code the freeze was taken from
              "config": current_config()}
    path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8", newline="\n")
    return frozen


def changed_keys(path: Path = FROZEN_PATH) -> list[str]:
    """Names of settings whose live value differs from the frozen one."""
    frozen = json.loads(path.read_text(encoding="utf-8")).get("config", {})
    current = current_config()
    return sorted(key for key in set(frozen) | set(current) if frozen.get(key) != current.get(key))


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze or verify the system config.")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        if not FROZEN_PATH.exists():
            raise SystemExit("not frozen: frozen_config.json does not exist")
        changed = changed_keys()
        print("frozen config", "OK" if not changed else f"CHANGED: {', '.join(changed)}")
        raise SystemExit(1 if changed else 0)
    frozen = write_freeze()
    print(f"frozen at {frozen['frozen_at']} from commit {frozen['git_commit']}")
    for key, value in frozen["config"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
