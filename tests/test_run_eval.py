import pytest

from benchmark import run_eval
from benchmark.judge import JudgeError
from src import config as cfg
from src.generate import GenerationError

A_URL, B_URL = "https://catalog.utk.edu/a", "https://catalog.utk.edu/b"
QUESTIONS = [
    {"id": "q1", "question": "hours for COSC 311?", "expected_answer": "3",
     "source_urls": [A_URL], "category": "single_course"},
    {"id": "q2", "question": "who teaches COSC 311?", "expected_answer": "Not in the catalog.",
     "source_urls": [], "category": "unanswerable"},
]


def chunk(cid, url):
    return {"id": cid, "url": url, "title": cid, "section": "", "text": cid}


class FakeRetriever:
    """bm25 finds page A first; vector finds it second."""

    def search(self, query, mode, k):
        hits = [chunk("a", A_URL), chunk("b", B_URL)]
        return hits if mode == "bm25" else hits[::-1]


def answer_fn(question, chunks):
    return {"answer": "3 hours [1].", "citations": [], "model": "claude-opus-5-5",
            "usage": {"input_tokens": 1000, "output_tokens": 100}, "latency_s": 2.0, "stop_reason": "end_turn"}


def judge_fn(question, response_text):
    label = "correct_refusal" if question["category"] == "unanswerable" else "correct"
    return {"notes": "", "label": label, "cited": "no", "citation_correct": "na",
            "judge_model": "claude-sonnet-5-5", "usage": {"input_tokens": 2000, "output_tokens": 50}}


def test_parse_chunking_reads_strategy_and_size_and_rejects_junk():
    assert run_eval.parse_chunking("packed:300") == ("packed", 300)
    for bad in ("packed", "weird:300", "section:many"):
        with pytest.raises(ValueError):
            run_eval.parse_chunking(bad)


def test_index_paths_use_the_shipped_index_only_for_the_default_chunking():
    default = run_eval.index_paths(cfg.CHUNK_STRATEGY, cfg.CHUNK_MAX_WORDS)
    other = run_eval.index_paths("packed", 300)
    assert default["chroma_dir"] == cfg.CHROMA_DIR and default["bm25_path"] == cfg.BM25_PATH
    assert cfg.EVAL_INDEX_DIR in other["chroma_dir"].parents
    assert other["chroma_dir"] != default["chroma_dir"]


def test_dev_split_is_always_allowed(tmp_path):
    run_eval.check_split_allowed("dev", verify=lambda: False, frozen_path=tmp_path / "none.json")


def test_test_split_refuses_when_the_hash_does_not_match(tmp_path):
    frozen = tmp_path / "frozen.json"
    frozen.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        run_eval.check_split_allowed("test", verify=lambda: False, frozen_path=frozen)


def test_test_split_refuses_until_the_config_is_frozen(tmp_path):
    with pytest.raises(SystemExit):
        run_eval.check_split_allowed("test", verify=lambda: True, frozen_path=tmp_path / "none.json")


def test_retrieve_all_records_rank_recall_and_chunk_ids_per_mode():
    rows = run_eval.retrieve_all(QUESTIONS, FakeRetriever(), modes=("bm25", "vector"), k=2)
    by = {(r["id"], r["mode"]): r for r in rows}
    assert len(rows) == 4
    assert by[("q1", "bm25")]["rank"] == 1 and by[("q1", "vector")]["rank"] == 2
    assert by[("q1", "bm25")]["chunk_ids"] == ["a", "b"] and by[("q1", "bm25")]["recall"] == 1.0
    assert by[("q2", "bm25")]["rank"] is None and by[("q2", "bm25")]["recall"] is None


def test_context_fn_changes_what_is_sent_but_not_how_search_is_scored():
    def only_page_a(question, hits):
        return [h for h in hits if h["url"] == A_URL]

    rows = run_eval.retrieve_all(QUESTIONS, FakeRetriever(), modes=("vector",), k=2, context_fn=only_page_a)
    q1 = rows[0]
    assert q1["chunk_ids"] == ["b", "a"] and q1["rank"] == 2      # search result, untouched
    assert q1["context_ids"] == ["a"] and q1["context_recall"] == 1.0
    assert [c["id"] for c in q1["chunks"]] == ["a"]               # this is what gets generated from


def test_summarize_retrieval_scores_answerable_questions_only():
    rows = run_eval.retrieve_all(QUESTIONS, FakeRetriever(), modes=("bm25", "vector"), k=2)
    summary = run_eval.summarize_retrieval(rows)
    assert summary["bm25"]["n"] == 1 and summary["bm25"]["hit@1"] == 1.0
    assert summary["vector"]["hit@1"] == 0.0 and summary["vector"]["mrr"] == 0.5


def test_answer_all_grades_every_row_and_reuses_answers_for_identical_context():
    class SameEveryMode(FakeRetriever):
        def search(self, query, mode, k):
            return [chunk("a", A_URL)]

    calls = []

    def counting_answer(question, chunks):
        calls.append(question)
        return answer_fn(question, chunks)

    rows = run_eval.retrieve_all(QUESTIONS, SameEveryMode(), modes=("bm25", "vector"), k=1)
    graded = run_eval.answer_all(rows, QUESTIONS, counting_answer, judge_fn, workers=1)
    assert len(calls) == 2                      # 2 questions, not 4 rows
    assert [r["grade"]["label"] for r in graded if r["id"] == "q1"] == ["correct", "correct"]
    assert all(r["error"] is None for r in graded)
    assert all("chunks" in r for r in graded)   # input rows are not mutated or stripped


def test_answer_all_records_failures_without_stopping_the_run():
    def failing_answer(question, chunks):
        if "who teaches" in question:
            raise GenerationError("rate_limit", "slow down", retryable=True)
        return answer_fn(question, chunks)

    def failing_judge(question, response_text):
        raise JudgeError("cut off")

    rows = run_eval.retrieve_all(QUESTIONS, FakeRetriever(), modes=("bm25",), k=2)
    graded = {r["id"]: r for r in run_eval.answer_all(rows, QUESTIONS, failing_answer, failing_judge, workers=1)}
    assert "rate_limit" in graded["q2"]["error"] and "grade" not in graded["q2"]
    assert "judge" in graded["q1"]["error"] and graded["q1"]["answer"] == "3 hours [1]."


def test_summarize_answers_reports_metrics_failures_latency_and_cost():
    rows = run_eval.retrieve_all(QUESTIONS, FakeRetriever(), modes=("bm25",), k=2)

    def harsh_judge(question, response_text):
        grade = judge_fn(question, response_text)
        return {**grade, "label": "hallucinated"} if question["id"] == "q2" else grade

    graded = run_eval.answer_all(rows, QUESTIONS, answer_fn, harsh_judge, workers=1)
    out = run_eval.summarize_answers(graded)["bm25"]
    assert out["n"] == 2 and out["ungraded"] == 0
    assert out["accuracy"] == 0.5 and out["hallucination_rate"] == 0.5
    assert out["failure_groups"] == {"hallucination": 1}
    assert out["by_category"]["single_course"] == {"n": 1, "accuracy": 1.0}
    assert out["mean_generation_s"] == 2.0
    # 1000 in + 100 out on Opus at $4 / $20 per million tokens
    assert out["mean_generation_cost_usd"] == pytest.approx(0.006)


def save_run(tmp_path, graded, fingerprint):
    import json
    (tmp_path / "run.json").write_text(json.dumps({"config": fingerprint}), encoding="utf-8")
    lines = [json.dumps({**r, "chunking": "section:250"}) for r in run_eval.public_rows(graded)]
    (tmp_path / "rows.jsonl").write_text("\n".join(lines), encoding="utf-8")


def test_reused_answers_are_not_generated_or_charged_again(tmp_path):
    fingerprint = {key: "same" for key in run_eval.REUSE_FINGERPRINT}
    rows = run_eval.retrieve_all(QUESTIONS, FakeRetriever(), modes=("bm25",), k=2)
    save_run(tmp_path, run_eval.answer_all(rows, QUESTIONS, answer_fn, judge_fn, workers=1), fingerprint)

    def must_not_run(question, chunks):
        raise AssertionError("answer was generated again")

    cached = run_eval.load_reusable(tmp_path, fingerprint)["section:250"]
    again = run_eval.answer_all(rows, QUESTIONS, must_not_run, judge_fn, workers=1, cached=cached)
    assert [r["grade"]["label"] for r in again] == ["correct", "correct_refusal"]
    assert all(r["reused"] for r in again)
    assert run_eval.total_spend_usd(again) == 0.0


def test_reuse_skips_failed_rows_so_they_are_tried_again(tmp_path):
    def failing_judge(question, response_text):
        raise JudgeError("cut off")

    fingerprint = {key: "same" for key in run_eval.REUSE_FINGERPRINT}
    rows = run_eval.retrieve_all(QUESTIONS, FakeRetriever(), modes=("bm25",), k=2)
    save_run(tmp_path, run_eval.answer_all(rows, QUESTIONS, answer_fn, failing_judge, workers=1), fingerprint)
    assert run_eval.load_reusable(tmp_path, fingerprint) == {}


def test_reuse_refuses_when_the_prompt_or_models_changed(tmp_path):
    fingerprint = {key: "same" for key in run_eval.REUSE_FINGERPRINT}
    save_run(tmp_path, [], fingerprint)
    with pytest.raises(SystemExit):
        run_eval.load_reusable(tmp_path, {**fingerprint, "system_prompt_sha256": "edited"})


def test_public_rows_drop_the_chunk_texts_but_keep_ids():
    rows = run_eval.retrieve_all(QUESTIONS, FakeRetriever(), modes=("bm25",), k=2)
    public = run_eval.public_rows(rows)
    assert "chunks" not in public[0] and public[0]["chunk_ids"] == ["a", "b"]
