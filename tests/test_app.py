import pytest
from fastapi.testclient import TestClient

from app.main import RateLimiter, create_app
from src import config as cfg
from src.generate import GenerationError

URL = "https://catalog.utk.edu/preview_course_nopop.php?catoid=56&coid=1"
RESULT = {
    "answer": "COSC 360 is 4 credit hours [1].",
    "citations": [{"n": 1, "title": "COSC 360 - Systems Programming", "section": "", "url": URL}],
    "chunks": [{"id": "course:1", "title": "COSC 360 - Systems Programming", "section": "", "url": URL,
                "text": "COSC 360 - Systems Programming\nCredit Hours: 4", "added_by": "search",
                "score": 9.1, "rank": 1, "body": "internal field"}],
    "model": "claude-opus-5-5", "latency_s": 2.5, "mode": "hybrid_rerank", "stop_reason": "end_turn",
    "usage": {"input_tokens": 1000, "output_tokens": 50},
}


class FakePipeline:
    def __init__(self, error=None):
        self.calls, self.error = [], error

    def __call__(self, question, mode):
        self.calls.append((question, mode))
        if self.error:
            raise self.error
        return {**RESULT, "mode": mode}


def make_client(pipeline=None, **limits):
    pipeline = pipeline or FakePipeline()
    limiter = RateLimiter(per_client=limits.get("per_client", 100), window_s=60, daily_cap=limits.get("daily_cap", 1000))
    return TestClient(create_app(answer_fn=pipeline, limiter=limiter), raise_server_exceptions=False), pipeline


def test_health_reports_status_without_any_secret():
    client, _ = make_client()
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["catalog"] == cfg.CATALOG_NAME
    assert body["default_mode"] == cfg.RETRIEVAL_MODE and "hybrid_rerank" in body["modes"]
    assert "sk-ant" not in str(body)


def test_ask_returns_answer_citations_and_retrieved_sources():
    client, pipeline = make_client()
    r = client.post("/ask", json={"question": "  how many hours is COSC 360?  "})
    body = r.json()
    assert r.status_code == 200 and body["success"] is True and body["error"] is None
    data = body["data"]
    assert data["answer"] == RESULT["answer"]
    assert data["citations"] == RESULT["citations"]
    assert data["sources"] == [{"n": 1, "id": "course:1", "title": "COSC 360 - Systems Programming", "section": "",
                                "url": URL, "text": RESULT["chunks"][0]["text"], "added_by": "search"}]
    assert data["mode"] == cfg.RETRIEVAL_MODE and data["model"] == "claude-opus-5-5"
    assert pipeline.calls == [("how many hours is COSC 360?", cfg.RETRIEVAL_MODE)]     # trimmed, default mode
    assert "usage" not in data and "body" not in str(data["sources"])                  # internals stay inside


def test_ask_passes_a_chosen_retrieval_mode():
    client, pipeline = make_client()
    r = client.post("/ask", json={"question": "prereqs for COSC 360", "mode": "bm25"})
    assert r.status_code == 200 and r.json()["data"]["mode"] == "bm25"
    assert pipeline.calls[0][1] == "bm25"


@pytest.mark.parametrize("payload", [
    {"question": ""},
    {"question": "   \n  "},
    {"question": "x" * (cfg.API_MAX_QUESTION_CHARS + 1)},
    {"question": "ok", "mode": "telepathy"},
    {"mode": "bm25"},
    {"question": 42},
    {"question": "ok", "extra": "field"},
])
def test_bad_input_is_rejected_before_any_model_call(payload):
    client, pipeline = make_client()
    r = client.post("/ask", json=payload)
    body = r.json()
    assert r.status_code == 422 and body["success"] is False and body["data"] is None
    assert body["error"]["kind"] == "invalid_request" and body["error"]["message"]
    assert pipeline.calls == []


def test_a_body_that_is_not_json_is_rejected_cleanly():
    client, pipeline = make_client()
    r = client.post("/ask", content=b"not json", headers={"content-type": "application/json"})
    assert r.status_code == 422 and r.json()["success"] is False and pipeline.calls == []


def test_rate_limit_returns_429_with_retry_after_and_stops_calling_the_model():
    client, pipeline = make_client(per_client=2)
    codes = [client.post("/ask", json={"question": "q"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429] and len(pipeline.calls) == 2
    r = client.post("/ask", json={"question": "q"})
    assert int(r.headers["retry-after"]) >= 1
    assert r.json()["error"]["kind"] == "rate_limited"


def test_invalid_requests_do_not_use_up_the_rate_limit():
    client, pipeline = make_client(per_client=1)
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "q"}).status_code == 200


def test_limiter_is_per_client_and_forgets_after_the_window():
    now = [0.0]
    limiter = RateLimiter(per_client=2, window_s=60, daily_cap=100, clock=lambda: now[0])
    assert [limiter.check("a")[0] for _ in range(3)] == [True, True, False]
    assert limiter.check("b")[0] is True                      # another client is unaffected
    allowed, retry_after = limiter.check("a")
    assert allowed is False and 0 < retry_after <= 60
    now[0] = 61.0
    assert limiter.check("a")[0] is True


def test_limiter_daily_cap_protects_the_api_budget_across_all_clients():
    now = [0.0]
    limiter = RateLimiter(per_client=100, window_s=60, daily_cap=3, clock=lambda: now[0])
    assert [limiter.check(f"client{i}")[0] for i in range(4)] == [True, True, True, False]
    now[0] = 24 * 3600 + 1
    assert limiter.check("client9")[0] is True


@pytest.mark.parametrize("kind, retryable", [("rate_limit", True), ("server", True), ("network", True),
                                              ("auth", False), ("billing", False), ("bad_request", False)])
def test_model_failures_become_a_503_with_a_safe_message(kind, retryable):
    error = GenerationError(kind, "Check ANTHROPIC_API_KEY in .env. secret-detail", retryable=retryable)
    client, _ = make_client(FakePipeline(error=error))
    r = client.post("/ask", json={"question": "q"})
    body = r.json()
    assert r.status_code == 503 and body["success"] is False
    assert body["error"]["kind"] == "unavailable" and body["error"]["retryable"] is retryable
    assert ".env" not in str(body) and "secret-detail" not in str(body) and "ANTHROPIC" not in str(body)


def test_a_model_refusal_is_reported_as_such():
    client, _ = make_client(FakePipeline(error=GenerationError("refusal", "The model declined to answer.")))
    r = client.post("/ask", json={"question": "q"})
    assert r.status_code == 422 and r.json()["error"]["kind"] == "declined"


def test_an_unexpected_crash_is_a_generic_500_without_internals():
    client, _ = make_client(FakePipeline(error=RuntimeError("indexes do not match chunks.jsonl at C:\\secret\\path")))
    r = client.post("/ask", json={"question": "q"})
    assert r.status_code == 500 and r.json()["error"]["kind"] == "internal"
    assert "secret" not in r.text and "Traceback" not in r.text


def test_home_page_and_its_assets_are_served_with_security_headers():
    client, _ = make_client()
    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert 'id="question"' in page.text and "/static/app.js" in page.text
    assert "<script>" not in page.text                                   # no inline script: CSP stays strict
    assert client.get("/static/app.js").status_code == 200 and client.get("/static/style.css").status_code == 200
    assert page.headers["x-content-type-options"] == "nosniff"
    assert page.headers["cache-control"] == "no-cache"                   # an edited page is never served stale
    assert client.get("/static/app.js").headers["cache-control"] == "no-cache"
    assert "default-src 'self'" in page.headers["content-security-policy"]


def test_only_post_is_allowed_on_ask():
    client, _ = make_client()
    assert client.get("/ask").status_code == 405
