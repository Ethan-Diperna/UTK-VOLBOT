"""Wayback Machine helpers.

Used for two things only:
1. Discovering which course/program ids exist for this catalog (CDX index).
2. Fetching program pages, because the live catalog puts an AWS WAF
   JavaScript challenge on content.php and preview_program.php (observed
   2026-10-30, HTTP 202 + `x-amzn-waf-action: challenge`). Course pages
   (preview_course_nopop.php) are served live without a challenge.
Every archived fetch records its snapshot timestamp so staleness is explicit.
"""
from __future__ import annotations

import gzip
import re
import time

import httpx

from src.config import USER_AGENT

CDX_URL = "https://web.archive.org/cdx/search/cdx"
AVAILABLE_URL = "https://archive.org/wayback/available"
TIMEOUT = 60.0
MAX_RETRIES = 6
RETRY_BASE_SECONDS = 10.0  # archive.org answers 429 when polled too fast


def _get(url: str, params: dict | None = None) -> httpx.Response:
    """GET with retry on 429/5xx. Honors Retry-After when present."""
    with _client() as c:
        for attempt in range(MAX_RETRIES):
            r = c.get(url, params=params)
            if r.status_code < 400:
                return r
            if r.status_code != 429 and r.status_code < 500:
                r.raise_for_status()
            wait = float(r.headers.get("retry-after") or RETRY_BASE_SECONDS * (attempt + 1))
            print(f"    wayback {r.status_code}; waiting {wait:.0f}s", flush=True)
            time.sleep(wait)
        r.raise_for_status()
        return r


def _client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT,
                        follow_redirects=True)


def list_snapshot_urls(path: str, catoid: int) -> list[str]:
    """Return distinct original URLs archived for `path` and this catalog."""
    params = {
        "url": f"catalog.utk.edu/{path}*",
        "filter": [f"original:.*catoid={catoid}.*", "statuscode:200"],
        "collapse": "urlkey",
        "fl": "original",
    }
    r = _get(CDX_URL, params=params)
    return [line.strip() for line in r.text.splitlines() if line.strip()]


def closest_snapshot(url: str) -> tuple[str, str] | None:
    """Return (timestamp 'YYYYMMDDhhmmss', archive_url) or None."""
    r = _get(AVAILABLE_URL, params={"url": url})
    closest = r.json().get("archived_snapshots", {}).get("closest")
    if not closest or not closest.get("available"):
        return None
    return closest["timestamp"], closest["url"]


def fetch_snapshot(url: str) -> tuple[str, str] | None:
    """Fetch the raw archived HTML for `url`. Returns (timestamp, html)."""
    found = closest_snapshot(url)
    if not found:
        return None
    timestamp, archive_url = found
    # `id_` returns the original bytes without the Wayback toolbar injected.
    raw_url = archive_url.replace(f"/web/{timestamp}/", f"/web/{timestamp}id_/")
    body = _get(raw_url).content
    if body[:2] == b"\x1f\x8b":  # some snapshots are stored gzip-compressed
        body = gzip.decompress(body)
    return timestamp, body.decode("utf-8", errors="ignore")


def looks_like_catalog_page(html: str) -> bool:
    """Reject WAF challenge pages and empty snapshots."""
    return bool(re.search(r'id=["\'](?:acalog-content|acalog-page-title)["\']|course_preview_title', html))
