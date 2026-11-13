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
# The scraper tells the site who is asking. Set SCRAPER_CONTACT in .env to an
# address you can be reached at; no personal address is kept in the code.
# Without it, the project page is given as the contact.
SCRAPER_CONTACT = os.getenv("SCRAPER_CONTACT", "https://github.com/Ethan-Diperna/UTK-VOLBOT")
AGENT_NAME = "utk-catalog-rag/0.1 (UTK student course-catalog research project; contact {contact})"
USER_AGENT = AGENT_NAME.format(contact=SCRAPER_CONTACT)

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
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
CHROMA_DIR = DATA_DIR / "chroma"            # persistent vector index (gitignored)
BM25_PATH = DATA_DIR / "bm25_tokens.json"   # tokenized chunks for keyword search
CHROMA_COLLECTION = "catalog_chunks"

# ---- Chunking (Phase 3). Tuned on the dev set only in Phase 6. ----
# "section": one chunk per program-page section (split further only if too big).
# "packed": neighbouring sections merged up to CHUNK_MAX_WORDS, with overlap.
CHUNK_STRATEGY = "section"
# The embedding model reads at most 512 tokens. Measured on 2026-11-12: with
# 250 words the longest chunk is 317 tokens, so nothing is cut off. index.py
# prints the longest chunk on every build; keep it under 512 when tuning.
CHUNK_MAX_WORDS = 250
CHUNK_OVERLAP_LINES = 2      # lines repeated when one long section is split
CHUNK_OVERLAP_SECTIONS = 1   # sections repeated between "packed" chunks

# ---- Retrieval (Phase 4). The final mode is chosen on the dev set in Phase 6. ----
RETRIEVAL_MODE = "hybrid_rerank"   # vector | bm25 | hybrid | hybrid_rerank
TOP_K = 5                          # chunks handed to the LLM
RETRIEVAL_CANDIDATES = 20          # depth of each search before fusion / reranking
# RRF constant from the original paper (Cormack, Clarke, Buettcher 2009). A
# larger value flattens the difference between rank 1 and rank 10.
RRF_K = 60

# ---- Context building (Phase 6), see src/context.py. Chosen on the dev set. ----
# Steps applied to the search hits before they go to the model. Empty = send
# the top-k hits as they are. Options: named_courses, program_pages, prerequisites.
# All three on since 2026-11-12: dev accuracy 0.70 -> 0.84 (see benchmark/TUNING_LOG.md).
CONTEXT_STEPS: tuple[str, ...] = ("named_courses", "program_pages", "prerequisites")
CONTEXT_MAX_PROGRAM_PAGES = 2    # whole program pages sent per question (largest is ~2,300 words)
CONTEXT_PREREQ_MAX_HOPS = 4      # how far a prerequisite chain is followed
CONTEXT_PREREQ_MAX_CHUNKS = 8    # cap on courses added by following prerequisites

# ---- Models (names are config, not hardcoded in logic) ----
# Model ids verified against the claude-api skill reference on 2026-10-30.
# Generation uses the current Opus; the LLM judge uses the cheaper current
# Sonnet. Override either in .env.
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "claude-opus-5-5")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-5-5")
# Generation settings (Phase 5), checked against the claude-api reference on
# 2026-11-12. Claude Opus 5.5 always thinks; "effort" is the only depth control
# and "medium" is that model's default. Tuned on dev only in Phase 6.
GENERATION_EFFORT = os.getenv("GENERATION_EFFORT", "medium")   # low | medium | high | xhigh | max
GENERATION_MAX_TOKENS = 16000   # ceiling, not a target; answers are a few sentences
# If a safety classifier declines a request, the API re-runs it on a fallback
# model instead of failing. The model that actually answered is recorded on
# every result. Set to "" to turn this off.
GENERATION_FALLBACK_BETA = "server-side-fallback-2026-07-01"
# ---- Evaluation (Phase 6) ----
# US dollars per million tokens as (input, output), from the claude-api
# reference on 2026-11-12. Used only to estimate cost per query; a model that
# is not listed gets no cost estimate instead of a made-up one.
MODEL_PRICES_PER_MTOK = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
}
JUDGE_MAX_TOKENS = 4000      # a grade is a few lines of JSON plus hidden thinking
EVAL_WORKERS = 4             # parallel answer+judge calls during an eval run
EVAL_INDEX_DIR = DATA_DIR / "eval_indexes"       # indexes for non-default chunking (gitignored)
RESULTS_DIR = ROOT / "benchmark" / "results"     # one folder per eval run
# ---- Web app (Phase 8) ----
# Every question costs API money (about 4 cents), so a public demo needs
# limits. Both can be raised or lowered in .env without touching code.
API_MAX_QUESTION_CHARS = 500
API_RATE_WINDOW_SECONDS = 60
API_RATE_LIMIT_PER_CLIENT = int(os.getenv("API_RATE_LIMIT_PER_CLIENT", "5"))   # questions per window per IP
API_DAILY_REQUEST_CAP = int(os.getenv("API_DAILY_REQUEST_CAP", "200"))         # all clients together, per 24 h
# Chat memory (added after the benchmark, see src/followup.py). The page sends
# the last few turns with a follow-up; a small, cheap model rewrites the
# follow-up into a standalone question before the benchmarked pipeline runs.
API_MAX_HISTORY_TURNS = 3
API_MAX_HISTORY_ANSWER_CHARS = 4000   # per earlier answer; the page trims longer ones
REWRITE_MODEL = os.getenv("REWRITE_MODEL", "claude-haiku-5-5")
REWRITE_MAX_TOKENS = 1000    # one line of output; headroom in case the model thinks first
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# The bge-small-en-v1.5 model card (read 2026-11-12) says to prepend this
# instruction to short queries, and nothing to passages.
EMBEDDING_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---- Baseline system under comparison (decided 2026-10-30) ----
BASELINE_NAME = "UT Verse AI Assistant"
BASELINE_URL = "https://utverse.tennessee.edu/"
# Terms: UT policy IT0002 (Acceptable Use of IT Resources, eff. 2025-01-01).
# No automation clause; we collect baseline answers by hand only.
BASELINE_TERMS_URL = "https://policy.tennessee.edu/policy/it0002-acceptable-use-of-information-technology-resources/"
