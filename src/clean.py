"""Turn raw Acalog HTML into clean documents: data/processed/docs.jsonl.

  python -m src.clean            # writes docs.jsonl and prints stats + samples

Course page layout (preview_course_nopop.php, verified 2026-10-30):
  <h1 id='course_preview_title'>COSC 302 - Fundamental Algorithms</h1>
  <strong>4</strong> <em>Credit Hours</em> <hr>
  Description text...<br><br>
  <em>(RE) Prerequisite(s):</em> COSC 202 ...<br>
  <em>(RE) Corequisite(s):</em> ...<br>
Program page layout (preview_program.php):
  <h1 id="acalog-content">Program title</h1>
  <div class="program_description">...</div>
  <div class="acalog-core"><h2>Heading</h2> ... <li class="acalog-course">...</li></div>
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

from bs4 import BeautifulSoup, NavigableString, Tag

from src import config as cfg
from src.catalog_links import clean_text, course_prefix, split_code_title

COURSE_BODY_RE = re.compile(
    r"<h1 id='course_preview_title'>(?P<title>.*?)</h1>\s*"
    r"(?:<strong>(?P<hours>[^<]*)</strong>\s*<em>Credit Hours</em>\s*)?"
    r"(?:<hr>)?(?P<body>.*?)</p>",
    re.S,
)
LABEL_RE = re.compile(
    r"^\s*(?:<em>\s*</em>\s*)*<em>(?:<strong>)?(?P<label>[^<:]+):(?:</strong>)?</em>\s*(?P<value>.*)$",
    re.S)


def read_manifest_ok() -> dict[tuple[str, int], dict]:
    """Latest successful fetch per (kind, id)."""
    latest: dict[tuple[str, int], dict] = {}
    with cfg.MANIFEST_PATH.open(encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            if m["status"] == 200:
                latest[(m["kind"], m["id"])] = m
    return latest


# ----------------------------------------------------------------- courses
def parse_course_fields(body_html: str) -> tuple[str, dict[str, str]]:
    """Split the course body on <br>; first run is the description, then
    '<em>Label:</em> value' lines become fields."""
    segments = re.split(r"<br\s*/?>", body_html)
    description_parts: list[str] = []
    fields: dict[str, str] = {}
    current_label: str | None = None
    for seg in segments:
        m = LABEL_RE.match(seg)
        if m:
            current_label = clean_text(m.group("label"))
            fields[current_label] = clean_text(m.group("value"))
        elif current_label and clean_text(seg):
            fields[current_label] = (fields[current_label] + " " + clean_text(seg)).strip()
        elif clean_text(seg):
            description_parts.append(clean_text(seg))
    return " ".join(description_parts).strip(), fields


def course_doc(coid: int, html: str, entry: dict) -> dict | None:
    m = COURSE_BODY_RE.search(html)
    if not m:
        return None
    code, title = split_code_title(clean_text(m.group("title")))
    if not code:
        return None
    hours = clean_text(m.group("hours") or "")
    description, fields = parse_course_fields(m.group("body"))
    prereq = " ".join(v for k, v in fields.items() if "Prerequisite" in k) or None
    coreq = " ".join(v for k, v in fields.items() if "Corequisite" in k) or None
    lines = [f"{code} - {title}", f"Credit Hours: {hours or 'not stated'}", description]
    lines += [f"{k}: {v}" for k, v in fields.items()]
    return {
        "id": f"course:{coid}",
        "url": entry["url"],
        "title": f"{code} - {title}",
        "doc_type": "course",
        "dept": course_prefix(code),
        "text": "\n".join(line for line in lines if line),
        "metadata": {
            "code": code, "course_title": title, "credit_hours": hours or None,
            "description": description, "fields": fields,
            "prerequisites": prereq, "corequisites": coreq, "sections": [],
            **provenance(entry),
        },
    }


# ---------------------------------------------------------------- programs
def node_text(node: Tag) -> str:
    """Text with one <li> per line, so requirement lists stay readable."""
    parts: list[str] = []
    for child in node.descendants:
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif child.name in ("li", "br", "p", "div", "h2", "h3", "h4"):
            parts.append("\n")
    lines = [re.sub(r"[ \t\xa0]+", " ", ln).strip() for ln in "".join(parts).splitlines()]
    return re.sub(r"\n{2,}", "\n", "\n".join(lines)).strip()


def program_sections(soup: BeautifulSoup) -> list[dict]:
    sections: list[dict] = []
    for core in soup.select("div.acalog-core"):
        heading = core.find(["h2", "h3", "h4"])
        level = int(heading.name[1]) if heading else 2
        title = clean_text(str(heading)) if heading else ""
        if heading:
            heading.extract()
        body = node_text(core)
        if title or body:
            sections.append({"level": level, "heading": title, "text": body})
    return sections


def program_doc(poid: int, html: str, entry: dict) -> dict | None:
    soup = BeautifulSoup(html, "html.parser")
    # Wayback snapshots use id="acalog-content"; the live site (Oct 2026) uses
    # id="acalog-page-title". Accept both.
    h1 = soup.find("h1", id="acalog-content") or soup.find("h1", id="acalog-page-title")
    if not h1:
        return None
    title = clean_text(str(h1))
    desc_node = soup.select_one("div.program_description")
    description = node_text(desc_node) if desc_node else ""
    sections = program_sections(soup)
    parts = [title, description]
    parts += [f"{'#' * s['level']} {s['heading']}\n{s['text']}".strip() for s in sections]
    return {
        "id": f"program:{poid}",
        "url": entry["url"],
        "title": title,
        "doc_type": "program",
        "dept": None,
        "text": "\n\n".join(p for p in parts if p).strip(),
        "metadata": {"description": description, "sections": sections, **provenance(entry)},
    }


def provenance(entry: dict) -> dict:
    return {"catalog_year": cfg.CATALOG_YEAR, "catoid": cfg.CATALOG_ID,
            "source": entry["source"], "fetched_at": entry["fetched_at"],
            "wayback_timestamp": entry.get("wayback_timestamp")}


# ------------------------------------------------------------------- main
def build_docs() -> list[dict]:
    docs: list[dict] = []
    for (kind, ident), entry in sorted(read_manifest_ok().items()):
        html = (cfg.ROOT / entry["path"]).read_text(encoding="utf-8")
        doc = course_doc(ident, html, entry) if kind == "course" else program_doc(ident, html, entry)
        if doc:
            docs.append(doc)
        else:
            print(f"WARN could not parse {kind} {ident}")
    return docs


def write_docs(docs: list[dict]) -> None:
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with cfg.DOCS_PATH.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def print_stats(docs: list[dict]) -> None:
    by_type = Counter(d["doc_type"] for d in docs)
    by_dept = Counter(d["dept"] for d in docs if d["doc_type"] == "course")
    by_src = Counter(d["metadata"]["source"] for d in docs)
    lengths = sorted(len(d["text"]) for d in docs)
    print(f"docs: {len(docs)} {dict(by_type)}; sources {dict(by_src)}")
    print(f"courses by dept: {dict(by_dept.most_common())}")
    if lengths:
        print(f"text chars: min {lengths[0]} median {lengths[len(lengths) // 2]} max {lengths[-1]}")
    n_prereq = sum(1 for d in docs if d["doc_type"] == "course" and d["metadata"]["prerequisites"])
    print(f"courses with a prerequisite field: {n_prereq}/{by_type.get('course', 0)}")


def print_samples(docs: list[dict], n: int = 2) -> None:
    for doc_type in ("course", "program"):
        for d in [x for x in docs if x["doc_type"] == doc_type][:n]:
            print("-" * 70)
            meta = {k: v for k, v in d.items() if k != "text"}
            print(json.dumps(meta, indent=1, ensure_ascii=False)[:1500])
            print("TEXT:", d["text"][:900])


def main() -> None:
    # No options, but parsing arguments means --help prints help instead of
    # rewriting docs.jsonl.
    argparse.ArgumentParser(description="Rebuild data/processed/docs.jsonl from the scraped pages in "
                                        "data/raw/, then print stats and samples.").parse_args()
    all_docs = build_docs()
    write_docs(all_docs)
    print_stats(all_docs)
    print_samples(all_docs)


if __name__ == "__main__":
    main()
