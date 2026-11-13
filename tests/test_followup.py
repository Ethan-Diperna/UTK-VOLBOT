from types import SimpleNamespace

import anthropic
import pytest

from src import config as cfg
from src.followup import build_rewrite_message, standalone_question

HISTORY = [{"question": "how many hours is cosc 360", "answer": "COSC 360 is 4 credit hours [1]."}]


class FakeClient:
    """Stands in for the Anthropic client: records requests, returns `reply`."""

    def __init__(self, reply="What are the prerequisites for COSC 360?", error=None):
        self.requests, self.reply, self.error = [], reply, error
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **request):
        self.requests.append(request)
        if self.error:
            raise self.error
        return SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking="hmm"),
                                        SimpleNamespace(type="text", text=self.reply)])


def test_a_first_question_is_returned_unchanged_without_any_model_call():
    client = FakeClient()

    result = standalone_question("what are its prereqs", [], client=client)

    assert result == "what are its prereqs" and client.requests == []


def test_a_follow_up_is_rewritten_using_the_conversation():
    client = FakeClient(reply="  What are the prerequisites for COSC 360?  ")

    result = standalone_question("what are its prereqs", HISTORY, client=client)

    assert result == "What are the prerequisites for COSC 360?"
    request = client.requests[0]
    assert request["model"] == cfg.REWRITE_MODEL and request["max_tokens"] == cfg.REWRITE_MAX_TOKENS
    sent = request["messages"][0]["content"]
    assert "how many hours is cosc 360" in sent and "COSC 360 is 4 credit hours [1]." in sent
    assert sent.rstrip().endswith("what are its prereqs")


def test_only_the_most_recent_turns_are_sent():
    history = [{"question": f"question {i}", "answer": f"answer {i}"} for i in range(cfg.API_MAX_HISTORY_TURNS + 2)]

    message = build_rewrite_message("and that one?", history)

    assert "question 0" not in message and "question 1" not in message
    assert f"question {cfg.API_MAX_HISTORY_TURNS + 1}" in message


@pytest.mark.parametrize("reply", ["", "   ", "line one\nline two", "x" * (cfg.API_MAX_QUESTION_CHARS + 1)])
def test_an_unusable_rewrite_falls_back_to_the_original_question(reply):
    client = FakeClient(reply=reply)

    assert standalone_question("what are its prereqs", HISTORY, client=client) == "what are its prereqs"


def test_a_failed_rewrite_call_falls_back_to_the_original_question():
    client = FakeClient(error=anthropic.AnthropicError("service down"))

    assert standalone_question("what are its prereqs", HISTORY, client=client) == "what are its prereqs"
