"""Web API and page for the catalog assistant.

  python -m uvicorn app.main:app --port 8000      # then open http://localhost:8000

  POST /ask     {"question": "...", "mode": "hybrid_rerank"}   mode is optional
  GET  /health  status, catalog year, whether the index and key are in place
  GET  /        the one-page frontend (app/static)

Every response from /ask has the same shape: {"success", "data", "error"}.
Each question costs API money, so /ask validates its input and is rate
limited before the model is ever called. Error messages sent to the browser
are fixed strings; the real cause is only written to the server log.
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Annotated, Callable, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, StringConstraints
from starlette.exceptions import HTTPException as StarletteHTTPException

from src import config as cfg
from src.context import build_context
from src.generate import GenerationError, generate
from src.retrieve import MODES, Retriever

log = logging.getLogger("catalog_app")
STATIC_DIR = Path(__file__).resolve().parent / "static"
SECONDS_PER_DAY = 24 * 60 * 60
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    # Scripts and styles only from this server: no inline code, no third parties.
    "Content-Security-Policy": "default-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
}
MESSAGES = {
    "busy": "The answer service is busy or unreachable right now. Please try again in a minute.",
    "down": "The answer service is not available right now.",
    "declined": "The assistant declined to answer that question. Try rephrasing it.",
    "internal": "Something went wrong on our side. Please try again.",
    "daily": "The demo has reached its daily question limit. Please come back tomorrow.",
}


class AskRequest(BaseModel):
    """Input for /ask. Unknown fields are rejected rather than ignored."""
    model_config = ConfigDict(extra="forbid")

    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1,
                                               max_length=cfg.API_MAX_QUESTION_CHARS)]
    mode: Literal[MODES] | None = None   # None means the configured default


class RateLimiter:
    """Two limits, both in memory. Per client: at most `per_client` questions
    in any `window_s` seconds (sliding window). Overall: at most `daily_cap`
    per 24 hours, which bounds the API bill even if many clients show up.

    Kept in one process's memory, so it resets on restart and is not shared
    between several server processes. That is enough for a single demo server."""

    def __init__(self, per_client: int, window_s: float, daily_cap: int,
                 clock: Callable[[], float] = time.monotonic):
        self.per_client, self.window_s, self.daily_cap, self.clock = per_client, window_s, daily_cap, clock
        self._recent: dict[str, deque[float]] = {}
        self._day_start = clock()
        self._day_count = 0
        self._lock = threading.Lock()

    def check(self, client: str) -> tuple[bool, int]:
        """(allowed, seconds to wait). An allowed call is counted."""
        with self._lock:
            now = self.clock()
            if now - self._day_start >= SECONDS_PER_DAY:
                self._day_start, self._day_count = now, 0
            recent = self._recent.setdefault(client, deque())
            while recent and recent[0] <= now - self.window_s:
                recent.popleft()
            if len(recent) >= self.per_client:
                return False, max(1, math.ceil(recent[0] + self.window_s - now))
            if self._day_count >= self.daily_cap:
                if not recent:
                    del self._recent[client]
                return False, max(1, math.ceil(self._day_start + SECONDS_PER_DAY - now))
            recent.append(now)
            self._day_count += 1
            return True, 0

    def daily_cap_reached(self) -> bool:
        with self._lock:
            return self._day_count >= self.daily_cap


def default_pipeline() -> Callable[[str, str], dict]:
    """The real question -> answer function. Indexes and models load on the
    first question, so the server starts fast and /health works without them.
    Search runs under a lock because the local models are not meant to be
    called from several threads at once; the slow Claude call runs outside it."""
    lock = threading.Lock()
    state: dict[str, Retriever] = {}

    def answer(question: str, mode: str) -> dict:
        with lock:
            if "retriever" not in state:
                state["retriever"] = Retriever.from_disk()
            retriever = state["retriever"]
            hits = retriever.search(question, mode=mode, k=cfg.TOP_K)
            chunks = build_context(question, hits, retriever.chunks)
        return {**generate(question, chunks), "mode": mode}

    return answer


def _failure(status: int, kind: str, message: str, headers: dict | None = None, **extra) -> JSONResponse:
    body = {"success": False, "data": None, "error": {"kind": kind, "message": message, **extra}}
    return JSONResponse(body, status_code=status, headers=headers)


def _public_result(result: dict, mode: str) -> dict:
    """Only what the page needs: no token counts, scores or internal fields."""
    sources = [{"n": n, "id": c["id"], "title": c["title"], "section": c.get("section", ""), "url": c["url"],
                "text": c["text"], "added_by": c.get("added_by", "search")}
               for n, c in enumerate(result["chunks"], start=1)]
    return {"answer": result["answer"], "citations": result["citations"], "sources": sources,
            "mode": mode, "model": result.get("model"), "latency_s": result.get("latency_s")}


def create_app(answer_fn: Callable[[str, str], dict] | None = None, limiter: RateLimiter | None = None) -> FastAPI:
    """Build the app. Tests pass a fake `answer_fn` and their own limiter."""
    answer = answer_fn or default_pipeline()
    limits = limiter or RateLimiter(cfg.API_RATE_LIMIT_PER_CLIENT, cfg.API_RATE_WINDOW_SECONDS,
                                    cfg.API_DAILY_REQUEST_CAP)
    app = FastAPI(title="UTK Catalog Assistant (unofficial)", docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        # "no-cache" = keep a copy but ask the server whether it changed before
        # using it. Without this, browsers keep showing an old page after an edit.
        response.headers.setdefault("Cache-Control", "no-cache")
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        first = exc.errors()[0]
        field = ".".join(str(part) for part in first["loc"] if part != "body") or "body"
        return _failure(422, "invalid_request", f"{field}: {first['msg']}")

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        return _failure(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(Exception)
    async def crash(request: Request, exc: Exception):
        log.exception("unhandled error on %s", request.url.path)
        return _failure(500, "internal", MESSAGES["internal"])

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok", "catalog": cfg.CATALOG_NAME, "default_mode": cfg.RETRIEVAL_MODE, "modes": list(MODES),
            "index_ready": cfg.CHUNKS_PATH.exists() and cfg.BM25_PATH.exists() and cfg.CHROMA_DIR.exists(),
            "api_key_configured": bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")),
        }

    @app.post("/ask")
    def ask(body: AskRequest, request: Request):
        # request.client.host is the direct peer. Behind a reverse proxy every
        # visitor shares the proxy's address unless uvicorn runs with
        # --proxy-headers; a client-sent X-Forwarded-For is never trusted here.
        client = request.client.host if request.client else "unknown"
        allowed, wait_s = limits.check(client)
        if not allowed:
            message = (MESSAGES["daily"] if limits.daily_cap_reached()
                       else f"Too many questions in a short time. Please wait {wait_s} seconds.")
            return _failure(429, "rate_limited", message, headers={"Retry-After": str(wait_s)})
        mode = body.mode or cfg.RETRIEVAL_MODE
        try:
            result = answer(body.question, mode)
        except GenerationError as e:
            log.warning("generation failed (%s): %s", e.kind, e)
            if e.kind == "refusal":
                return _failure(422, "declined", MESSAGES["declined"])
            return _failure(503, "unavailable", MESSAGES["busy" if e.retryable else "down"], retryable=e.retryable)
        return {"success": True, "data": _public_result(result, mode), "error": None}

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
