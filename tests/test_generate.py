from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from src import config as cfg
from src import generate

CHUNKS = [
    {"id": "course:1", "url": "https://catalog.utk.edu/c?coid=1", "title": "COSC 360 - Systems Programming",
     "section": "", "text": "COSC 360 - Systems Programming\nCredit Hours: 4"},
    {"id": "program:9#03", "url": "https://catalog.utk.edu/p?poid=9", "title": "Computer Science Major, BS",
     "section": "Term 4", "text": "Computer Science Major, BS\nSection: Term 4\nCOSC 360 - Systems Programming 4 Credit Hours"},
    {"id": "course:2", "url": "https://catalog.utk.edu/c?coid=2", "title": "COSC 367 - Honors: Systems Programming",
     "section": "", "text": "COSC 367 - Honors: Systems Programming\nCredit Hours: 4"},
]


def message(text="COSC 360 is 4 credit hours [1].", stop_reason="end_turn", model="claude-opus-5-5"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason, stop_details=None, model=model,
        usage=SimpleNamespace(input_tokens=321, output_tokens=12))


class FakeClient:
    """Stands in for anthropic.Anthropic: records the request, returns or raises."""

    def __init__(self, result=None, error=None):
        self.calls = []
        self._result = result or message()
        self._error = error
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return self._result


def api_error(cls, status):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("boom", response=httpx2.Response(status, request=request), body=None)


def test_user_message_numbers_excerpts_from_one_and_ends_with_the_question():
    text = generate.build_user_message("how many hrs is cosc360", CHUNKS)
    assert text.index("[1]") < text.index("[2]") < text.index("[3]")
    assert "https://catalog.utk.edu/c?coid=1" in text
    assert "Credit Hours: 4" in text and "Term 4" in text
    assert text.rstrip().endswith("how many hrs is cosc360")


def test_extract_citations_keeps_first_appearance_order_without_duplicates():
    cites = generate.extract_citations("Both are 4 hours [3][1]. Also [1].", CHUNKS)
    assert [c["n"] for c in cites] == [3, 1]
    assert cites[1] == {"n": 1, "title": CHUNKS[0]["title"], "section": "", "url": CHUNKS[0]["url"]}


def test_extract_citations_reads_comma_lists_and_ignores_numbers_out_of_range():
    cites = generate.extract_citations("See [1, 2] and [7] and [0].", CHUNKS)
    assert [c["n"] for c in cites] == [1, 2]


def test_generate_returns_answer_citations_chunks_and_usage():
    client = FakeClient()
    out = generate.generate("How many credit hours is COSC 360?", CHUNKS, client=client)
    assert out["answer"] == "COSC 360 is 4 credit hours [1]."
    assert out["citations"] == [{"n": 1, "title": CHUNKS[0]["title"], "section": "", "url": CHUNKS[0]["url"]}]
    assert out["chunks"] == CHUNKS
    assert out["model"] == "claude-opus-5-5"
    assert out["usage"] == {"input_tokens": 321, "output_tokens": 12}
    assert out["stop_reason"] == "end_turn"


def test_generate_sends_configured_model_system_prompt_and_excerpts():
    client = FakeClient()
    generate.generate("How many credit hours is COSC 360?", CHUNKS, client=client)
    call = client.calls[0]
    assert call["model"] == cfg.GENERATION_MODEL
    assert call["system"] == generate.SYSTEM_PROMPT
    assert call["output_config"] == {"effort": cfg.GENERATION_EFFORT}
    assert "thinking" not in call                     # always on for this model; a config would 400
    user = call["messages"][0]
    assert user["role"] == "user" and "Credit Hours: 4" in user["content"]


def test_system_prompt_states_the_three_rules():
    prompt = generate.SYSTEM_PROMPT.lower()
    assert "only" in prompt and "[1]" in prompt and "does not contain" in prompt


def test_generate_joins_only_text_blocks():
    result = message()
    result.content.append(SimpleNamespace(type="text", text=" It includes a lab [1]."))
    out = generate.generate("q", CHUNKS, client=FakeClient(result))
    assert out["answer"] == "COSC 360 is 4 credit hours [1]. It includes a lab [1]."


def test_no_chunks_answers_not_found_without_calling_the_api():
    client = FakeClient()
    out = generate.generate("who teaches cosc 360", [], client=client)
    assert client.calls == []
    assert out["answer"] == generate.NO_CONTEXT_ANSWER
    assert out["citations"] == [] and out["model"] is None


def test_empty_question_raises_value_error():
    with pytest.raises(ValueError):
        generate.generate("   ", CHUNKS, client=FakeClient())


def test_refusal_stop_reason_becomes_generation_error():
    with pytest.raises(generate.GenerationError) as e:
        generate.generate("q", CHUNKS, client=FakeClient(message(text="", stop_reason="refusal")))
    assert e.value.kind == "refusal"


def test_max_tokens_stop_reason_is_reported_not_hidden():
    out = generate.generate("q", CHUNKS, client=FakeClient(message(stop_reason="max_tokens")))
    assert out["stop_reason"] == "max_tokens"


@pytest.mark.parametrize("error, kind, retryable", [
    (api_error(anthropic.AuthenticationError, 401), "auth", False),
    (api_error(anthropic.PermissionDeniedError, 403), "auth", False),
    (api_error(anthropic.NotFoundError, 404), "bad_request", False),
    (api_error(anthropic.BadRequestError, 400), "bad_request", False),
    (api_error(anthropic.RateLimitError, 429), "rate_limit", True),
    (api_error(anthropic.InternalServerError, 500), "server", True),
    (anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com")), "network", True),
])
def test_api_errors_become_generation_errors_with_kind_and_plain_message(error, kind, retryable):
    with pytest.raises(generate.GenerationError) as e:
        generate.generate("q", CHUNKS, client=FakeClient(error=error))
    assert e.value.kind == kind
    assert e.value.retryable is retryable
    assert "boom" not in str(e.value)                  # no raw API text leaks to the user
    assert e.value.__cause__ is error


def test_missing_api_key_is_a_clean_auth_error_before_any_request(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    generate.get_client.cache_clear()
    with pytest.raises(generate.GenerationError) as e:
        generate.generate("q", CHUNKS)          # no client passed: uses the real one
    assert e.value.kind == "auth" and ".env" in str(e.value)
    generate.get_client.cache_clear()


def test_answer_question_retrieves_then_generates():
    class FakeRetriever:
        def search(self, query, mode, k):
            self.seen = (query, mode, k)
            return CHUNKS[:2]

    retriever, client = FakeRetriever(), FakeClient()
    out = generate.answer_question("How many credit hours is COSC 360?", retriever,
                                   mode="hybrid", k=2, client=client)
    assert retriever.seen == ("How many credit hours is COSC 360?", "hybrid", 2)
    assert out["mode"] == "hybrid" and len(out["chunks"]) == 2
    assert out["citations"][0]["url"] == CHUNKS[0]["url"]
