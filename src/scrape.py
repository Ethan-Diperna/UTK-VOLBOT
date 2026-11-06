"""Scrape the UTK undergraduate catalog (Acalog) to data/raw/.

Commands
  python -m src.scrape discover [--wayback]       build data/raw/course_index.json
  python -m src.scrape programs [--all] [--source wayback|live]
  python -m src.scrape courses  [--prefix COSC MATH ...] [--all] [--limit N]
                                [--delay SECONDS] [--source live|wayback]
  python -m src.scrape import-saved                 register pages you saved from a browser
  python -m src.scrape status

Design notes (verified 2026-10-30):
* Live course pages (preview_course_nopop.php) are served normally.
* Live listing pages (content.php) and program pages (preview_program.php)
  return HTTP 202 with an AWS WAF JavaScript challenge to non-browser
  clients. We do not try to defeat that. Program pages default to Wayback
  Machine snapshots, with the snapshot timestamp recorded in the manifest.
* robots.txt sets crawl-delay 120 for all bots. Default delay honors it.
  `--delay` exists so you can make a different, explicit choice.
* Course discovery (which coids exist) comes from: hand-saved listing pages
  in data/raw/listings/, course links on fetched program pages, prerequisite
  links on fetched course pages, and (optional) the Wayback CDX index.
* Everything is resumable: pages already on disk with status 200 are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from src import config as cfg
from src import wayback
from src.catalog_links import (CourseLink, clean_text, course_prefix,
                               parse_course_links, parse_program_links,
                               split_code_title)

WAYBACK_COURSES_DIR = cfg.RAW_DIR / "wayback" / "courses"
WAYBACK_FETCH_DELAY = 2.0  # polite to archive.org; it rate-limits bursts
LIVE_TIMEOUT = 30.0
TITLE_RE = re.compile(r"<h1 id='course_preview_title'>(.*?)</h1>", re.S)


# ----------------------------------------------------------------- manifest
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_manifest(entry: dict) -> None:
    cfg.MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with cfg.MANIFEST_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_manifest() -> list[dict]:
    if not cfg.MANIFEST_PATH.exists():
        return []
    with cfg.MANIFEST_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fetched_ok(kind: str) -> set[int]:
    return {m["id"] for m in read_manifest() if m["kind"] == kind and m["status"] == 200}


def save_page(path: Path, html: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return hashlib.sha256(html.encode("utf-8")).hexdigest()


def course_url(coid: int) -> str:
    return f"{cfg.CATALOG_BASE_URL}/preview_course_nopop.php?catoid={cfg.CATALOG_ID}&coid={coid}"


def program_url(poid: int) -> str:
    return f"{cfg.CATALOG_BASE_URL}/preview_program.php?catoid={cfg.CATALOG_ID}&poid={poid}"


# -------------------------------------------------------------- live fetch
class LiveFetcher:
    """One HTTP client, fixed delay between requests, WAF-challenge aware."""

    def __init__(self, delay: float):
        self.delay = delay
        self.client = httpx.Client(headers={"User-Agent": cfg.USER_AGENT},
                                   timeout=LIVE_TIMEOUT, follow_redirects=True)
        self._last = 0.0

    def get(self, url: str) -> tuple[int, str]:
        wait = self.delay - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        try:
            r = self.client.get(url)
        finally:
            self._last = time.monotonic()
        if r.headers.get("x-amzn-waf-action") == "challenge":
            return 202, ""
        return r.status_code, r.text


def fetch_one(url: str, source: str, live: LiveFetcher | None) -> tuple[int, str, str | None]:
    """Return (status, html, wayback_timestamp). 202 means WAF challenge."""
    if source == "live":
        status, html = live.get(url)
        return status, html, None
    got = wayback.fetch_snapshot(url)
    time.sleep(WAYBACK_FETCH_DELAY)
    if not got:
        return 404, "", None
    ts, html = got
    if not wayback.looks_like_catalog_page(html):
        return 202, "", ts  # archived copy is a WAF challenge page, not content
    return 200, html, ts


# ----------------------------------------------------------- course index
def load_index() -> dict:
    if cfg.COURSE_INDEX_PATH.exists():
        return json.loads(cfg.COURSE_INDEX_PATH.read_text(encoding="utf-8"))
    return {"catoid": cfg.CATALOG_ID, "catalog_year": cfg.CATALOG_YEAR,
            "updated_at": None, "courses": {}}


def save_index(index: dict) -> None:
    index["updated_at"] = now_iso()
    cfg.COURSE_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg.COURSE_INDEX_PATH.write_text(json.dumps(index, indent=1, ensure_ascii=False),
                                     encoding="utf-8")


def add_links(index: dict, links: list[CourseLink], source: str) -> int:
    """Merge links into the index. Existing entries are replaced, not mutated."""
    added = 0
    courses = index["courses"]
    for link in links:
        if link.catoid != cfg.CATALOG_ID:
            continue
        key = str(link.coid)
        old = courses.get(key, {"code": None, "title": None, "sources": []})
        if key not in courses:
            added += 1
        courses[key] = {
            "code": old["code"] or link.code,
            "title": old["title"] or link.title,
            "sources": sorted(set(old["sources"]) | {source}),
        }
    return added


def self_link_from_page(coid: int, html: str) -> list[CourseLink]:
    """A course page's <h1> is 'CODE - Title'; turn it into a CourseLink."""
    m = TITLE_RE.search(html)
    if not m:
        return []
    code, title = split_code_title(clean_text(m.group(1)))
    return [CourseLink(cfg.CATALOG_ID, coid, code, title)]


def _html_files(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.html")) if folder.exists() else []


def discover(use_wayback: bool) -> None:
    index = load_index()
    for p in _html_files(cfg.RAW_LISTINGS_DIR):
        n = add_links(index, parse_course_links(p.read_text(encoding="utf-8")), f"listing:{p.name}")
        print(f"listing {p.name}: +{n}")
    for p in _html_files(cfg.RAW_PROGRAMS_DIR):
        n = add_links(index, parse_course_links(p.read_text(encoding="utf-8")), f"program:{p.stem}")
        print(f"program {p.stem}: +{n}")
    for p in _html_files(cfg.RAW_COURSES_DIR):
        html = p.read_text(encoding="utf-8")
        add_links(index, self_link_from_page(int(p.stem), html), f"course:{p.stem}")
        add_links(index, parse_course_links(html), f"course:{p.stem}")
    if use_wayback:
        discover_from_wayback(index)
    save_index(index)
    known = sum(1 for c in index["courses"].values() if c["code"])
    print(f"index: {len(index['courses'])} coids, {known} with a course code")


def discover_from_wayback(index: dict) -> None:
    """Resolve archived course pages to (coid, code, title). Off UTK servers."""
    urls = wayback.list_snapshot_urls("preview_course_nopop.php", cfg.CATALOG_ID)
    coids = sorted({int(u.split("coid=")[1].split("&")[0]) for u in urls if "coid=" in u})
    print(f"wayback cdx: {len(coids)} archived course pages")
    for coid in coids:
        if index["courses"].get(str(coid), {}).get("code"):
            continue
        path = WAYBACK_COURSES_DIR / f"{coid}.html"
        if not path.exists():
            got = wayback.fetch_snapshot(course_url(coid))
            time.sleep(WAYBACK_FETCH_DELAY)
            if not got or not wayback.looks_like_catalog_page(got[1]):
                continue
            save_page(path, got[1])
        add_links(index, self_link_from_page(coid, path.read_text(encoding="utf-8")), "wayback_cdx")
        save_index(index)


# --------------------------------------------------------------- programs
def wanted_program_ids(all_programs: bool) -> dict[int, str]:
    if not all_programs:
        return dict(cfg.DEFAULT_PROGRAM_IDS)
    ids: dict[int, str] = {}
    for navoid in (cfg.NAVOID_MAJORS, cfg.NAVOID_MINORS, cfg.NAVOID_CERTIFICATES):
        got = wayback.fetch_snapshot(
            f"{cfg.CATALOG_BASE_URL}/content.php?catoid={cfg.CATALOG_ID}&navoid={navoid}")
        if got:
            ids.update({p.poid: p.title for p in parse_program_links(got[1])})
    return ids


def fetch_programs(all_programs: bool, source: str, delay: float) -> None:
    done = fetched_ok("program")
    targets = {k: v for k, v in wanted_program_ids(all_programs).items() if k not in done}
    print(f"programs to fetch: {len(targets)} (already done: {len(done)})")
    live = LiveFetcher(delay) if source == "live" else None
    for poid, title in targets.items():
        url, path = program_url(poid), cfg.RAW_PROGRAMS_DIR / f"{poid}.html"
        status, html, ts = fetch_one(url, source, live)
        entry = {"kind": "program", "id": poid, "title": title, "url": url,
                 "path": str(path.relative_to(cfg.ROOT)), "source": source,
                 "fetched_at": now_iso(), "status": status, "sha256": None,
                 "wayback_timestamp": ts}
        if status == 200:
            entry["sha256"] = save_page(path, html)
        append_manifest(entry)
        print(f"  {status} {source} poid={poid} {title[:60]}", flush=True)


# ---------------------------------------------------------------- courses
def wanted_course_ids(prefixes: list[str] | None, all_courses: bool) -> list[int]:
    index = load_index()
    out = []
    for coid, c in index["courses"].items():
        if all_courses or (c["code"] and course_prefix(c["code"]) in (prefixes or [])):
            out.append(int(coid))
    return sorted(out)


def fetch_courses(prefixes: list[str] | None, all_courses: bool, limit: int | None,
                  delay: float, source: str) -> None:
    done = fetched_ok("course")
    targets = [c for c in wanted_course_ids(prefixes, all_courses) if c not in done]
    if limit:
        targets = targets[:limit]
    per = delay if source == "live" else WAYBACK_FETCH_DELAY
    print(f"courses to fetch: {len(targets)} (done: {len(done)}), "
          f"~{len(targets) * per / 60:.0f} min at {per}s delay", flush=True)
    live = LiveFetcher(delay) if source == "live" else None
    for coid in targets:
        url, path = course_url(coid), cfg.RAW_COURSES_DIR / f"{coid}.html"
        status, html, ts = fetch_one(url, source, live)
        entry = {"kind": "course", "id": coid, "url": url,
                 "path": str(path.relative_to(cfg.ROOT)), "source": source,
                 "fetched_at": now_iso(), "status": status, "sha256": None,
                 "wayback_timestamp": ts}
        index = load_index()  # reload each time so a parallel `discover` is not clobbered
        if status == 200:
            entry["sha256"] = save_page(path, html)
            add_links(index, self_link_from_page(coid, html), f"course:{coid}")
            add_links(index, parse_course_links(html), f"course:{coid}")  # prereq closure
            save_index(index)
        append_manifest(entry)
        code = index["courses"].get(str(coid), {}).get("code")
        print(f"  {status} {source} coid={coid} {code or ''}", flush=True)
        if status == 202 and source == "live":
            print("WAF challenge on a course page; stopping so we do not hammer it.")
            break


# ----------------------------------------------------------- hand-saved
def import_saved() -> None:
    """Register program/course pages saved by hand from a browser.

    Save the page as "Webpage, HTML only" to data/raw/programs/<poid>.html or
    data/raw/courses/<coid>.html. This records them in the manifest with
    source "browser_saved" so clean.py picks them up and provenance stays honest.
    """
    for kind, folder, make_url in (("program", cfg.RAW_PROGRAMS_DIR, program_url),
                                   ("course", cfg.RAW_COURSES_DIR, course_url)):
        done = fetched_ok(kind)
        for p in _html_files(folder):
            if not p.stem.isdigit() or int(p.stem) in done:
                continue
            html = p.read_text(encoding="utf-8", errors="ignore")
            if not wayback.looks_like_catalog_page(html):
                print(f"skip {p}: does not look like a catalog page")
                continue
            append_manifest({"kind": kind, "id": int(p.stem), "url": make_url(int(p.stem)),
                             "path": str(p.relative_to(cfg.ROOT)), "source": "browser_saved",
                             "fetched_at": now_iso(), "status": 200,
                             "sha256": hashlib.sha256(html.encode("utf-8")).hexdigest(),
                             "wayback_timestamp": None})
            print(f"registered {kind} {p.stem}")


# ----------------------------------------------------------------- status
def status() -> None:
    m = read_manifest()
    for kind in ("program", "course"):
        rows = [x for x in m if x["kind"] == kind]
        ok = [x for x in rows if x["status"] == 200]
        by_src: dict[str, int] = {}
        for x in ok:
            by_src[x["source"]] = by_src.get(x["source"], 0) + 1
        print(f"{kind}s: {len(ok)} ok / {len(rows)} attempts, by source {by_src}")
    index = load_index()
    prefixes: dict[str, int] = {}
    for c in index["courses"].values():
        if c["code"]:
            p = course_prefix(c["code"])
            prefixes[p] = prefixes.get(p, 0) + 1
    print(f"catalog: {cfg.CATALOG_NAME} (catoid={cfg.CATALOG_ID})")
    print(f"course index: {len(index['courses'])} coids; by prefix (top 15): "
          f"{dict(sorted(prefixes.items(), key=lambda kv: -kv[1])[:15])}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("discover")
    d.add_argument("--wayback", action="store_true", help="also resolve archived course pages")
    p = sub.add_parser("programs")
    p.add_argument("--all", action="store_true")
    p.add_argument("--source", choices=["wayback", "live"], default="wayback")
    p.add_argument("--delay", type=float, default=cfg.ROBOTS_CRAWL_DELAY_SECONDS)
    c = sub.add_parser("courses")
    c.add_argument("--prefix", nargs="*", default=cfg.DEFAULT_COURSE_PREFIXES)
    c.add_argument("--all", action="store_true")
    c.add_argument("--limit", type=int)
    c.add_argument("--delay", type=float, default=cfg.ROBOTS_CRAWL_DELAY_SECONDS)
    c.add_argument("--source", choices=["live", "wayback"], default="live")
    sub.add_parser("import-saved")
    sub.add_parser("status")
    a = ap.parse_args(argv)
    if a.cmd == "discover":
        discover(a.wayback)
    elif a.cmd == "programs":
        fetch_programs(a.all, a.source, a.delay)
    elif a.cmd == "courses":
        if a.source == "live" and a.delay < cfg.ROBOTS_CRAWL_DELAY_SECONDS:
            print(f"NOTE: delay {a.delay}s is below robots.txt crawl-delay "
                  f"{cfg.ROBOTS_CRAWL_DELAY_SECONDS}s.", file=sys.stderr)
        fetch_courses(a.prefix, a.all, a.limit, a.delay, a.source)
    elif a.cmd == "import-saved":
        import_saved()
    elif a.cmd == "status":
        status()


if __name__ == "__main__":
    main()
