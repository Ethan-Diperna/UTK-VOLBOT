import io
import json

import pytest

from benchmark import verify
from benchmark.questions import load

URL = "https://catalog.utk.edu/preview_course_nopop.php?catoid=56&coid=1"


def row(qid, **over):
    base = {"id": qid, "question": f"q {qid}", "expected_answer": "4",
            "source_urls": [URL], "category": "single_course", "origin": "self",
            "notes": "", "verified_by_hand": False}
    return {**base, **over}


def test_apply_decision_verify_marks_row_and_keeps_input_unchanged():
    rows = [row("a"), row("b")]
    out = verify.apply_decision(rows, "a", "y", now="2026-11-04T00:00:00+00:00")
    assert out[0]["verified_by_hand"] is True
    assert out[0]["verified_at"] == "2026-11-04T00:00:00+00:00"
    assert out[1]["verified_by_hand"] is False
    assert rows[0]["verified_by_hand"] is False


def test_apply_decision_edit_replaces_answer():
    out = verify.apply_decision([row("a")], "a", "e", "3 credit hours")
    assert out[0]["expected_answer"] == "3 credit hours"
    assert out[0]["verified_by_hand"] is True


def test_apply_decision_edit_rejects_empty_answer():
    with pytest.raises(ValueError):
        verify.apply_decision([row("a")], "a", "e", "   ")


def test_apply_decision_drop_removes_row():
    out = verify.apply_decision([row("a"), row("b")], "a", "d")
    assert [r["id"] for r in out] == ["b"]


def test_run_walks_only_unverified_rows_and_saves_each_decision(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [
        row("a", verified_by_hand=True), row("b"), row("c"), row("d"), row("e")]) + "\n",
        encoding="utf-8")
    stream = io.StringIO("y\ne\nnew answer line 1\nline 2\n.\nd\nq\n")
    out = io.StringIO()
    saved = verify.run(path, stream=stream, out=out, open_browser=False)
    assert saved == 3
    rows = {r["id"]: r for r in load(path)}
    assert rows["b"]["verified_by_hand"] is True
    assert rows["c"]["expected_answer"] == "new answer line 1\nline 2"
    assert "d" not in rows
    assert rows["e"]["verified_by_hand"] is False
    assert "[a]" not in out.getvalue()


def test_run_skip_leaves_row_unverified(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(row("a")) + "\n", encoding="utf-8")
    saved = verify.run(path, stream=io.StringIO("s\n"), out=io.StringIO(), open_browser=False)
    assert saved == 0
    assert load(path)[0]["verified_by_hand"] is False
