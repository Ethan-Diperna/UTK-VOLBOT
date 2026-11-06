"""Parse course/program links out of Acalog HTML.

Link shapes observed on catalog.utk.edu (2026-10-30):
  <a href="preview_course_nopop.php?catoid=56&coid=371380" ...>AC 101 - Intro...</a>
  <a href="preview_course_nopop.php?catoid=56&coid=370268" aria-label="View course details for CHEM 112*">CHEM 112*</a>
  <a href="preview_program.php?catoid=56&poid=36396" ...>Applied Artificial Intelligence Major, BS ...</a>
Program pages use a JS form instead of an href:
  <a href="#" aria-label="View course details for COSC 103 - Introduction to Computer Science"
     onClick="showCourse('56', '391301',this, ...)">COSC 103 - Introduction...</a>
"""
from __future__ import annotations

import html as htmllib
import re
from dataclasses import dataclass

# Course codes look like "COSC 302", "MATH 141", "ECE 313", "AI 201".
# A trailing "*" marks a Vol Core course on this site; it is not part of the code.
COURSE_CODE_RE = re.compile(r"\b([A-Z]{2,5})\s?(\d{3}[A-Z]?)\*?")

_COURSE_LINK_RE = re.compile(
    r'<a\s[^>]*href="[^"]*preview_course_nopop\.php\?catoid=(\d+)&(?:amp;)?coid=(\d+)"'
    r'[^>]*>(.*?)</a>',
    re.S,
)
_SHOWCOURSE_RE = re.compile(
    r'<a\s[^>]*aria-label="View course details for ([^"]*)"[^>]*'
    r"onClick=\"showCourse\('(\d+)',\s*'(\d+)'",
    re.S | re.I,
)
_PROGRAM_LINK_RE = re.compile(
    r'<a\s[^>]*href="[^"]*preview_program\.php\?catoid=(\d+)&(?:amp;)?poid=(\d+)"'
    r'[^>]*>(.*?)</a>',
    re.S,
)
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class CourseLink:
    catoid: int
    coid: int
    code: str | None   # "COSC 302" or None when the link text has no code
    title: str | None  # "Introduction to Algorithms" or None


@dataclass(frozen=True)
class ProgramLink:
    catoid: int
    poid: int
    title: str


def clean_text(fragment: str) -> str:
    text = htmllib.unescape(_TAG_RE.sub(" ", fragment))
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def split_code_title(text: str) -> tuple[str | None, str | None]:
    """'COSC 302* - Intro to Algorithms' -> ('COSC 302', 'Intro to Algorithms')."""
    m = COURSE_CODE_RE.match(text)
    if not m:
        return None, None
    code = f"{m.group(1)} {m.group(2)}"
    rest = text[m.end():].strip()
    title = re.sub(r"^[-–—:]\s*", "", rest).strip() or None
    return code, title


def parse_course_links(html: str) -> list[CourseLink]:
    seen: set[tuple[int, int]] = set()
    out: list[CourseLink] = []
    for catoid, coid, inner in _COURSE_LINK_RE.findall(html):
        key = (int(catoid), int(coid))
        if key in seen:
            continue
        seen.add(key)
        code, title = split_code_title(clean_text(inner))
        out.append(CourseLink(key[0], key[1], code, title))
    for label, catoid, coid in _SHOWCOURSE_RE.findall(html):
        key = (int(catoid), int(coid))
        if key in seen:
            continue
        seen.add(key)
        code, title = split_code_title(clean_text(label))
        out.append(CourseLink(key[0], key[1], code, title))
    return out


def parse_program_links(html: str) -> list[ProgramLink]:
    seen: set[tuple[int, int]] = set()
    out: list[ProgramLink] = []
    for catoid, poid, inner in _PROGRAM_LINK_RE.findall(html):
        key = (int(catoid), int(poid))
        title = clean_text(inner)
        if key in seen or not title:
            continue
        seen.add(key)
        out.append(ProgramLink(key[0], key[1], title))
    return out


def course_prefix(code: str) -> str:
    return code.split(" ")[0]
