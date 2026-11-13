import json
from types import SimpleNamespace

import pytest

from benchmark import judge
from src import config as cfg

QUESTION = {"id": "q1", "question": "How many credit hours is COSC 311?", "expected_answer": "3",
            "source_urls": ["https://catalog.utk.edu/c?coid=1"], "category": "single_course"}
GRADE = {"notes": "", "label": "correct", "cited": "yes", "citation_correct": "yes"}


class FakeClient:
    def __init__(self, payload=None, stop_reason="end_turn", text=None):
        self.calls = []
        body = text if text is not None else json.dumps(payload or GRADE)
        self._result = SimpleNamespace(
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=body)],
            stop_reason=stop_reason, model="claude-sonnet-5-5",
            usage=SimpleNamespace(input_tokens=2000, output_tokens=40))
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._result


def test_render_response_appends_the_cited_sources_a_user_would_see():
    cites = [{"n": 2, "title": "COSC 311 - Discrete Structures", "section": "", "url": "https://x/1"}]
    text = judge.render_response("It is 3 hours [2].", cites)
    assert text.startswith("It is 3 hours [2].")
    assert "[2] COSC 311 - Discrete Structures" in text and "https://x/1" in text
    assert judge.render_response("No idea.", []) == "No idea."


def test_judge_message_holds_question_expected_answer_sources_and_response():
    msg = judge.build_judge_message(QUESTION, "COSC 311 is 3 credit hours.",
                                    page_titles={"https://catalog.utk.edu/c?coid=1": "COSC 311 - Discrete Structures"})
    assert "How many credit hours is COSC 311?" in msg
    assert "Expected answer: 3" in msg
    assert "COSC 311 - Discrete Structures" in msg and "https://catalog.utk.edu/c?coid=1" in msg
    assert "Response A" in msg and "COSC 311 is 3 credit hours." in msg


def test_judge_message_never_reveals_the_category_or_the_system():
    msg = judge.build_judge_message(QUESTION, "x")
    assert "single_course" not in msg and "category" not in msg.lower()


def test_judge_message_for_a_question_without_source_pages_says_none():
    msg = judge.build_judge_message({**QUESTION, "source_urls": []}, "x")
    assert "(none)" in msg


def test_judge_sends_the_rubric_verbatim_as_system_prompt_and_asks_for_json():
    client = FakeClient()
    judge.judge(QUESTION, "COSC 311 is 3 credit hours.", client=client)
    call = client.calls[0]
    assert call["system"] == judge.rubric_text()
    assert call["model"] == cfg.JUDGE_MODEL
    schema = call["output_config"]["format"]["schema"]
    assert set(schema["properties"]["label"]["enum"]) == set(judge.LABELS)
    assert schema["additionalProperties"] is False


def test_judge_returns_grade_model_and_usage():
    out = judge.judge(QUESTION, "COSC 311 is 3 credit hours.", client=FakeClient())
    assert out["label"] == "correct" and out["cited"] == "yes" and out["citation_correct"] == "yes"
    assert out["judge_model"] == "claude-sonnet-5-5"
    assert out["usage"] == {"input_tokens": 2000, "output_tokens": 40}


@pytest.mark.parametrize("client", [
    FakeClient(stop_reason="max_tokens"),
    FakeClient(stop_reason="refusal"),
    FakeClient(text="not json"),
    FakeClient(payload={**GRADE, "label": "mostly_right"}),
])
def test_judge_raises_instead_of_guessing_a_grade(client):
    with pytest.raises(judge.JudgeError):
        judge.judge(QUESTION, "x", client=client)


def test_rubric_hash_is_a_sha256_of_the_rubric_file():
    assert len(judge.rubric_sha256()) == 64
