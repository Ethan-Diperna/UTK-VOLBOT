"""Evaluate retrieval and answers on a question split.

  python -m benchmark.run_eval --split dev                       # retrieval only, free
  python -m benchmark.run_eval --split dev --chunking section:250 packed:250
  python -m benchmark.run_eval --split dev --answers             # + Claude answers + LLM judge (costs money)
  python -m benchmark.run_eval --split dev --answers --limit 3   # cheap smoke run

Retrieval is scored on answerable questions: was a chunk from one of the
question's source pages retrieved (hit@1/3/5, MRR)? With --answers each
question is also answered and graded by the LLM judge using GRADING_RUBRIC.md.
Every run is saved under benchmark/results/ with its full config.

The test split is guarded: it runs only when questions_test.jsonl still
matches its locked hash AND benchmark/frozen_config.json exists, so the test
set cannot be touched by accident while the system is still being tuned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import anthropic

from benchmark import lock, metrics
from benchmark.judge import JudgeError, judge, render_response, rubric_sha256
from benchmark.questions import BENCH_DIR, DEV_PATH, TEST_PATH, load
from src import config as cfg
from src.chunk import STRATEGIES, load_jsonl
from src.context import STEPS as CONTEXT_STEPS
from src.context import build_context
from src.generate import SYSTEM_PROMPT, GenerationError, generate
from src.index import build_indexes
from src.retrieve import MODES, Retriever

FROZEN_PATH = BENCH_DIR / "frozen_config.json"
WARMUP_QUERY = "warm up the models"   # run once per mode so model loading is not timed
MS_PER_S = 1000
# What _answer_one produces for one question; these fields are what --reuse carries over.
RESULT_FIELDS = ("answer", "citations", "generation_model", "generation_usage", "generation_s",
                 "stop_reason", "grade", "error")
# Config values that must be unchanged for an earlier run's answers to be reusable.
REUSE_FINGERPRINT = ("system_prompt_sha256", "rubric_sha256", "generation_model",
                     "generation_effort", "judge_model", "top_k")


def parse_chunking(spec: str) -> tuple[str, int]:
    """'packed:300' -> ('packed', 300)."""
    strategy, sep, size = spec.partition(":")
    if not sep or strategy not in STRATEGIES or not size.isdigit():
        raise ValueError(f"bad chunking {spec!r}; expected strategy:max_words, e.g. section:250")
    return strategy, int(size)


def default_chunking() -> str:
    return f"{cfg.CHUNK_STRATEGY}:{cfg.CHUNK_MAX_WORDS}"


def index_paths(strategy: str, max_words: int) -> dict[str, Path]:
    """The default chunking is evaluated on the very index the app serves.
    Other setups get their own folder so they never overwrite it."""
    if (strategy, max_words) == (cfg.CHUNK_STRATEGY, cfg.CHUNK_MAX_WORDS):
        return {"chunks_path": cfg.CHUNKS_PATH, "bm25_path": cfg.BM25_PATH, "chroma_dir": cfg.CHROMA_DIR}
    base = cfg.EVAL_INDEX_DIR / f"{strategy}_{max_words}"
    return {"chunks_path": base / "chunks.jsonl", "bm25_path": base / "bm25_tokens.json",
            "chroma_dir": base / "chroma"}


def load_retriever(strategy: str, max_words: int, rebuild: bool = False) -> tuple[Retriever, dict]:
    paths = index_paths(strategy, max_words)
    is_default = paths["chroma_dir"] == cfg.CHROMA_DIR
    if not is_default and (rebuild or not paths["bm25_path"].exists()):
        build_indexes(strategy, max_words, **paths)
    retriever = Retriever.from_disk(**paths)
    meta = retriever.bm25.meta
    if (meta.get("strategy"), meta.get("max_words")) != (strategy, max_words):
        raise SystemExit(f"index at {paths['bm25_path']} was built as {meta.get('strategy')}:"
                         f"{meta.get('max_words')}, not {strategy}:{max_words}; run `python -m src.index`")
    return retriever, meta


def check_split_allowed(split: str, verify: Callable[[], bool] = lock.verify,
                        frozen_path: Path = FROZEN_PATH) -> None:
    if split != "test":
        return
    if not verify():
        raise SystemExit("questions_test.jsonl does not match test_set.sha256; refusing to run.")
    if not frozen_path.exists():
        raise SystemExit(f"{frozen_path.name} not found: the config is not frozen yet. "
                         "The test set is run once, after tuning on dev is finished.")


def retrieve_all(questions: list[dict], retriever, modes: tuple[str, ...], k: int,
                 context_fn: Callable[[str, list[dict]], list[dict]] | None = None) -> list[dict]:
    """One row per (question, mode). `rank` and `recall` score the ranked
    search hits; `context_recall` scores what is sent to the model after
    `context_fn` (src.context.build_context) has added to them. All three are
    None for unanswerable questions, which have no source page to find."""
    rows = []
    for mode in modes:
        retriever.search(WARMUP_QUERY, mode=mode, k=k)
        for q in questions:
            started = time.perf_counter()
            hits = retriever.search(q["question"], mode=mode, k=k)
            context = context_fn(q["question"], hits) if context_fn else hits
            elapsed_ms = (time.perf_counter() - started) * MS_PER_S
            urls = [c["url"] for c in hits]
            answerable = bool(q["source_urls"])
            rows.append({
                "id": q["id"], "category": q["category"], "mode": mode,
                "chunk_ids": [c["id"] for c in hits], "urls": urls,
                "rank": metrics.first_hit_rank(urls, q["source_urls"]) if answerable else None,
                "recall": metrics.source_recall(urls, q["source_urls"]) if answerable else None,
                "context_ids": [c["id"] for c in context],
                "context_recall": (metrics.source_recall([c["url"] for c in context], q["source_urls"])
                                   if answerable else None),
                "retrieval_ms": round(elapsed_ms, 1), "chunks": context,
            })
    return rows


def _by_mode(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for r in rows:
        grouped.setdefault(r["mode"], []).append(r)
    return grouped


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize_retrieval(rows: list[dict]) -> dict[str, dict]:
    out = {}
    for mode, mode_rows in _by_mode(rows).items():
        answerable = [r for r in mode_rows if r["recall"] is not None]
        out[mode] = {**metrics.retrieval_metrics([r["rank"] for r in answerable]),
                     "source_recall": _mean([r["recall"] for r in answerable]),
                     "context_recall": _mean([r["context_recall"] for r in answerable]),
                     "context_chunks": _mean([len(r["context_ids"]) for r in mode_rows]),
                     "median_ms": statistics.median(r["retrieval_ms"] for r in mode_rows)}
    return out


def _answer_one(q: dict, chunks: list[dict], answer_fn, judge_fn) -> dict:
    try:
        out = answer_fn(q["question"], chunks)
    except GenerationError as e:
        return {"error": f"generation {e.kind}: {e}"}
    result = {"answer": out["answer"], "citations": out["citations"], "generation_model": out["model"],
              "generation_usage": out["usage"], "generation_s": out["latency_s"],
              "stop_reason": out["stop_reason"]}
    try:
        grade = judge_fn(q, render_response(out["answer"], out["citations"]))
    except (JudgeError, anthropic.APIError) as e:
        return {**result, "error": f"judge: {e}"}
    return {**result, "grade": grade, "error": None}


def load_reusable(run_dir: Path, current: dict) -> dict[str, dict]:
    """Graded answers from an earlier run, as {chunking: {(question id, chunk
    ids): result}}, so a run that died half way (rate limit, no credits) does
    not pay twice. Refuses unless the earlier run used the same prompt, rubric,
    models and effort, because otherwise its answers are not comparable."""
    earlier = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["config"]
    changed = [key for key in REUSE_FINGERPRINT if earlier.get(key) != current.get(key)]
    if changed:
        raise SystemExit(f"cannot reuse {run_dir.name}: {', '.join(changed)} changed since that run")
    reusable: dict[str, dict] = {}
    for r in load_jsonl(run_dir / "rows.jsonl"):
        if r.get("grade") and r.get("error") is None:
            result = {key: r[key] for key in RESULT_FIELDS}
            sent = r.get("context_ids", r["chunk_ids"])   # runs before context steps had no context_ids
            reusable.setdefault(r["chunking"], {})[(r["id"], tuple(sent))] = result
    return reusable


def answer_all(rows: list[dict], questions: list[dict], answer_fn, judge_fn,
               workers: int = cfg.EVAL_WORKERS, cached: dict | None = None) -> list[dict]:
    """Answer and grade every row. Two modes that retrieved the same chunks in
    the same order for a question would send Claude the identical prompt, so
    that answer is generated and judged once and shared: it saves money and
    keeps model randomness from showing up as a difference between modes.
    `cached` holds results from load_reusable; those are not paid for again."""
    by_id = {q["id"]: q for q in questions}
    cached = cached or {}
    jobs = {(r["id"], tuple(r["context_ids"])): r["chunks"] for r in rows
            if (r["id"], tuple(r["context_ids"])) not in cached}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        done = pool.map(lambda item: _answer_one(by_id[item[0][0]], item[1], answer_fn, judge_fn),
                        jobs.items())
        results = {**{key: {**value, "reused": True} for key, value in cached.items()},
                   **{key: {**value, "reused": False} for key, value in zip(jobs, done)}}
    graded = []
    for r in rows:
        result = results[(r["id"], tuple(r["context_ids"]))]
        group = (metrics.failure_group(r["category"], result["grade"]["label"], r["rank"] is not None)
                 if result.get("grade") else None)
        graded.append({**r, **result, "failure_group": group})
    return graded


def _cost(usage: dict | None, model: str | None) -> float | None:
    if not usage:
        return None
    return metrics.cost_usd(usage["input_tokens"], usage["output_tokens"],
                            cfg.MODEL_PRICES_PER_MTOK.get(model))


def summarize_answers(rows: list[dict]) -> dict[str, dict]:
    out = {}
    for mode, mode_rows in _by_mode(rows).items():
        graded = [r for r in mode_rows if r.get("grade")]
        categories = sorted({r["category"] for r in graded})
        gen_costs = [c for r in graded if (c := _cost(r["generation_usage"], r["generation_model"])) is not None]
        judge_costs = [c for r in graded
                       if (c := _cost(r["grade"]["usage"], r["grade"]["judge_model"])) is not None]
        out[mode] = {
            **metrics.answer_metrics([r["grade"] for r in graded]),
            "ungraded": len(mode_rows) - len(graded),
            "failure_groups": dict(Counter(r["failure_group"] for r in graded if r["failure_group"])),
            "by_category": {cat: {key: metrics.answer_metrics(
                [r["grade"] for r in graded if r["category"] == cat])[key] for key in ("n", "accuracy")}
                for cat in categories},
            "mean_generation_s": _mean([r["generation_s"] for r in graded]),
            "mean_generation_cost_usd": _mean(gen_costs),
            "mean_judge_cost_usd": _mean(judge_costs),
        }
    return out


def total_spend_usd(rows: list[dict]) -> float:
    """Dollars actually spent: each shared answer is counted once."""
    unique = {(r.get("chunking"), r["id"], tuple(r["context_ids"])): r for r in rows
              if not r.get("reused")}   # reused answers were paid for by the earlier run
    spent = 0.0
    for r in unique.values():
        spent += _cost(r.get("generation_usage"), r.get("generation_model")) or 0.0
        if r.get("grade"):
            spent += _cost(r["grade"]["usage"], r["grade"]["judge_model"]) or 0.0
    return spent


def public_rows(rows: list[dict]) -> list[dict]:
    """Rows for saving: chunk ids stay, the bulky chunk texts do not."""
    return [{key: value for key, value in r.items() if key != "chunks"} for r in rows]


def _fmt(value) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def format_table(summaries: dict[str, dict[str, dict]], columns: tuple[str, ...]) -> str:
    """`summaries` is {chunking: {mode: metrics}}."""
    lines = [f"{'chunking':<13}{'mode':<15}" + "".join(f"{c:>20}" for c in columns)]
    for chunking, by_mode in summaries.items():
        for mode, m in by_mode.items():
            lines.append(f"{chunking:<13}{mode:<15}" + "".join(f"{_fmt(m.get(c)):>20}" for c in columns))
    return "\n".join(lines)


def _git_state() -> dict:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=cfg.ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def run_config(a: argparse.Namespace, questions_path: Path, n_questions: int, index_meta: dict) -> dict:
    return {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": a.split, "n_questions": n_questions, "limit": a.limit,
        "questions_sha256": hashlib.sha256(questions_path.read_bytes()).hexdigest(),
        "modes": list(a.modes), "chunking": index_meta, "top_k": a.k,
        "context_steps": list(a.context), "context_max_program_pages": cfg.CONTEXT_MAX_PROGRAM_PAGES,
        "context_prereq_max_hops": cfg.CONTEXT_PREREQ_MAX_HOPS,
        "context_prereq_max_chunks": cfg.CONTEXT_PREREQ_MAX_CHUNKS,
        "retrieval_candidates": cfg.RETRIEVAL_CANDIDATES, "rrf_k": cfg.RRF_K,
        "embedding_model": cfg.EMBEDDING_MODEL, "reranker_model": cfg.RERANKER_MODEL,
        "answers": a.answers, "generation_model": cfg.GENERATION_MODEL,
        "generation_effort": cfg.GENERATION_EFFORT, "judge_model": cfg.JUDGE_MODEL,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "rubric_sha256": rubric_sha256(), "git": _git_state(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate retrieval (and answers) on a question split.")
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    ap.add_argument("--chunking", nargs="+", default=[default_chunking()], metavar="STRATEGY:MAX_WORDS")
    ap.add_argument("--k", type=int, default=cfg.TOP_K)
    ap.add_argument("--answers", action="store_true", help="also generate and judge answers (API cost)")
    ap.add_argument("--limit", type=int, help="only the first N questions (smoke run)")
    ap.add_argument("--rebuild", action="store_true", help="rebuild non-default eval indexes")
    ap.add_argument("--context", nargs="*", choices=CONTEXT_STEPS, default=list(cfg.CONTEXT_STEPS),
                    help="context steps applied to the search hits (none = hits as they are)")
    ap.add_argument("--reuse", type=Path, metavar="RUN_DIR",
                    help="reuse graded answers from an earlier run folder instead of paying again")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    check_split_allowed(a.split)

    questions_path = DEV_PATH if a.split == "dev" else TEST_PATH
    questions = load(questions_path)[:a.limit]
    titles = {d["url"]: d["title"] for d in load_jsonl(cfg.DOCS_PATH)}
    retrieval, answers, index_meta, saved_rows = {}, {}, {}, []
    reusable = load_reusable(a.reuse, run_config(a, questions_path, len(questions), {})) if a.reuse else {}
    for spec in a.chunking:
        retriever, index_meta[spec] = load_retriever(*parse_chunking(spec), rebuild=a.rebuild)
        rows = retrieve_all(questions, retriever, tuple(a.modes), a.k,
                            context_fn=lambda question, hits, all_chunks=retriever.chunks: build_context(
                                question, hits, all_chunks, steps=tuple(a.context)))
        retrieval[spec] = summarize_retrieval(rows)
        if a.answers:
            rows = answer_all(rows, questions, generate,
                              lambda q, text: judge(q, text, page_titles=titles),
                              cached=reusable.get(spec))
            answers[spec] = summarize_answers(rows)
        saved_rows += [{**r, "chunking": spec} for r in public_rows(rows)]

    print(f"\nRetrieval on {a.split} ({len(questions)} questions, answerable ones scored):")
    print(f"context steps: {a.context or 'none'}")
    print(format_table(retrieval, ("n", "hit@1", "hit@3", "hit@5", "mrr", "source_recall",
                                   "context_recall", "context_chunks", "median_ms")))
    spend = total_spend_usd(saved_rows)
    if a.answers:
        print("\nAnswers, graded by the LLM judge:")
        print(format_table(answers, ("n", "ungraded", "accuracy", "partial_accuracy", "hallucination_rate",
                                     "citation_accuracy", "mean_generation_s")))
        print(f"\nAPI spend this run: ${spend:.2f}")

    out_dir = cfg.RESULTS_DIR / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{a.split}"
    out_dir.mkdir(parents=True, exist_ok=False)
    summary = {"config": run_config(a, questions_path, len(questions), index_meta),
               "retrieval": retrieval, "answers": answers, "api_spend_usd": round(spend, 4)}
    (out_dir / "run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8", newline="\n")
    with (out_dir / "rows.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in saved_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nsaved -> {out_dir}")


if __name__ == "__main__":
    main()
