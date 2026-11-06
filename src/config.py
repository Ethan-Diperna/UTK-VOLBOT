"""Central configuration. Nothing here is secret; secrets come from .env."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# ---- Catalog source (verified 2026-10-30 by inspecting catalog.utk.edu) ----
CATALOG_BASE_URL = "https://catalog.utk.edu"
# catoid=56 is the "2026-2027 Undergraduate Catalog" (current, not archived).
CATALOG_ID = 56
CATALOG_YEAR = "2026-2027"
CATALOG_NAME = "2026-2027 Undergraduate Catalog"

# Acalog navigation ids for listing pages (from the site nav bar).
NAVOID_COURSES = 12117       # "Course Descriptions"
NAVOID_MAJORS = 12035        # "Majors, A-Z"
NAVOID_MINORS = 12113        # "Minors, A-Z"
NAVOID_CERTIFICATES = 12114  # "Undergraduate Certificates A-Z"

# robots.txt (fetched 2026-10-30) says `crawl-delay: 120` for all bots and
# disallows /ajax/, /portfolio.php, /search_advanced.php. We honor it.
ROBOTS_CRAWL_DELAY_SECONDS = 120
USER_AGENT = (
    "utk-catalog-rag/0.1 (UTK student course-catalog research project; "
    "contact dipernaem@gmail.com)"
)

# Limited default scope. `--all` in scrape.py overrides.
DEFAULT_COURSE_PREFIXES = ["COSC", "MATH", "ECE", "AI", "DATA"]
# poids found on the archived "Majors, A-Z" and "Certificates A-Z" pages.
DEFAULT_PROGRAM_IDS = {
    36396: "Applied Artificial Intelligence Major, BS",
    36491: "Applied AI Major, BS - Applied Computing Concentration",
    36497: "Applied AI Major, BS - Technical Communication Concentration",
    36501: "Applied AI Major, BS - Leadership Studies Concentration",
    36373: "Applied Artificial Intelligence Certificate",
    35734: "Computer Science Major, BS",
    36397: "Data Science Major, BS",
    36381: "Data Science Certificate",
    35741: "Electrical Engineering Major, BS",
    35820: "Mathematics Major, BS",
    36068: "Mathematics Major, BS - Applied Math Concentration",
}

# ---- Paths ----
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_COURSES_DIR = RAW_DIR / "courses"
RAW_PROGRAMS_DIR = RAW_DIR / "programs"
RAW_LISTINGS_DIR = RAW_DIR / "listings"   # hand-saved listing pages go here
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_PATH = RAW_DIR / "manifest.jsonl"
COURSE_INDEX_PATH = RAW_DIR / "course_index.json"
DOCS_PATH = PROCESSED_DIR / "docs.jsonl"

# ---- Models (names are config, not hardcoded in logic) ----
# Model ids verified against the claude-api skill reference on 2026-10-30.
# Generation uses the current Opus; the LLM judge uses the cheaper current
# Sonnet. Override either in .env.
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "claude-opus-5-5")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-5-5")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---- Baseline system under comparison (decided 2026-10-30) ----
BASELINE_NAME = "UT Verse AI Assistant"
BASELINE_URL = "https://utverse.tennessee.edu/"
# Terms: UT policy IT0002 (Acceptable Use of IT Resources, eff. 2025-01-01).
# No automation clause; we collect baseline answers by hand only.
BASELINE_TERMS_URL = "https://policy.tennessee.edu/policy/it0002-acceptable-use-of-information-technology-resources/"
