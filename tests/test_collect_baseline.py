import io
import json

from benchmark import collect_baseline as cb


def dummy_questions(tmp_path):
    qp = tmp_path / "q.jsonl"
    rows = [{"id": f"q00{i}", "question": f"question {i}?", "expected_answer": "x",
             "source_urls": ["https://catalog.utk.edu/x"], "category": "single_course",
             "origin": "self", "notes": ""} for i in (1, 2, 3)]
    qp.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return qp


def test_skip_save_quit_and_resume(tmp_path):
    qp = dummy_questions(tmp_path)
    out_path = tmp_path / "dev.jsonl"
    # q001 skipped, q002 answered (two lines), then quit before q003
    stdin = io.StringIO("s\nline one\nline two\n.\nq\n")
    saved = cb.run("dev", "default chat", stream=stdin, out=io.StringIO(),
                   questions_path=qp, out_path=out_path)
    assert saved == 1
    rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rows] == ["q002"]
    assert rows[0]["response"] == "line one\nline two"
    assert rows[0]["question"] == "question 2?"
    assert rows[0]["system"] == "UT Verse AI Assistant"
    assert rows[0]["assistant_variant"] == "default chat"
    assert rows[0]["collected_at"].endswith("+00:00")
    # resume: q002 is skipped automatically; answer q001 and q003
    stdin = io.StringIO("a1\n.\na3\n.\n")
    saved = cb.run("dev", "default chat", stream=stdin, out=io.StringIO(),
                   questions_path=qp, out_path=out_path)
    assert saved == 2
    rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rows] == ["q002", "q001", "q003"]


def test_empty_response_not_saved(tmp_path):
    qp = dummy_questions(tmp_path)
    out_path = tmp_path / "dev.jsonl"
    stdin = io.StringIO(".\nq\n")
    assert cb.run("dev", "v", stream=stdin, out=io.StringIO(),
                  questions_path=qp, out_path=out_path) == 0
    assert not out_path.exists()
