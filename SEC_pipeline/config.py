"""
Shared configuration for the SEC 10-K pipeline (2018).

SEC EDGAR requires a User-Agent header identifying the requester.
Set SEC_USER_AGENT in your environment or .env file, e.g.:
    SEC_USER_AGENT="YourName your.email@example.com"
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Target year
# ──────────────────────────────────────────────────────────────

START_YEAR = 2018
END_YEAR = 2025
YEAR = END_YEAR  # backward compat: scripts that import YEAR get the end year
QUARTERS = [1, 2, 3, 4]

# ──────────────────────────────────────────────────────────────
# Directories (relative to SEC_pipeline/)
# ──────────────────────────────────────────────────────────────

PIPELINE_DIR = Path(__file__).parent
OUTPUT_DIR = PIPELINE_DIR / "output"
INDEX_DIR = OUTPUT_DIR / "index"
FINANCIALS_DIR = OUTPUT_DIR / "financials_json_2018"
RAW_HTML_DIR = OUTPUT_DIR / "10k_html_raw"
SANITIZED_HTML_DIR = OUTPUT_DIR / "10k_html_sanitized"
LLM_SANITIZED_DIR = OUTPUT_DIR / "10k_markdown"
TAGGED_DIR = OUTPUT_DIR / "10k_markdown_tagged"
WORK_DIR = OUTPUT_DIR / "_work"


def financials_dir_for_year(year: int) -> Path:
    """Directory for per-company financials JSON files for a given base year."""
    return OUTPUT_DIR / f"financials_json_{year}"


def raw_html_dir_for_year(year: int) -> Path:
    """Directory for raw 10-K HTML filings filed during a given year."""
    return OUTPUT_DIR / f"10k_html_raw_{year}"


def markdown_dir_for_year(year: int) -> Path:
    """Directory for sanitized 10-K markdown for a given filing year."""
    return OUTPUT_DIR / f"10k_markdown_{year}"


def index_csv_for_year(year: int) -> Path:
    """Path to the 10-K filing index CSV for a given year."""
    return INDEX_DIR / f"10k_filings_{year}.csv"

# ──────────────────────────────────────────────────────────────
# SEC EDGAR URLs
# ──────────────────────────────────────────────────────────────

EDGAR_BASE = "https://www.sec.gov"
EDGAR_FULL_INDEX = f"{EDGAR_BASE}/Archives/edgar/full-index"
EDGAR_ARCHIVES = f"{EDGAR_BASE}/Archives/edgar/data"

# SEC Financial Statement Data Sets (pre-parsed XBRL)
DERA_BASE = f"{EDGAR_BASE}/files/dera/data/financial-statement-data-sets"
DERA_ZIPS = [
    f"{DERA_BASE}/{y}q{q}.zip"
    for y in range(START_YEAR, END_YEAR + 1)
    for q in QUARTERS
]

# EDGAR full-text index (company.idx) per quarter
FULL_INDEX_URLS = [
    f"{EDGAR_FULL_INDEX}/{y}/QTR{q}/company.idx"
    for y in range(START_YEAR, END_YEAR + 1)
    for q in QUARTERS
]

# ──────────────────────────────────────────────────────────────
# SEC rate-limiting & headers
# ──────────────────────────────────────────────────────────────

# SEC allows max 10 requests/second. We stay conservative.
REQUEST_DELAY = 0.12  # seconds between requests (~8 req/s)
DOWNLOAD_CHUNK_SIZE = 1 * 1024 * 1024  # 1 MB


def get_sec_user_agent() -> str:
    """Load SEC_USER_AGENT from environment or .env file.

    SEC requires: 'Company/Person email@example.com'
    """
    agent = os.environ.get("SEC_USER_AGENT")
    if agent:
        return agent.strip()

    for env_path in [PIPELINE_DIR.parent / ".env", Path.cwd() / ".env"]:
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("SEC_USER_AGENT"):
                    _, _, val = line.partition("=")
                    return val.strip().strip('"').strip("'")

    raise RuntimeError(
        "SEC_USER_AGENT not found. Set it in your environment or .env file.\n"
        'Example: SEC_USER_AGENT="YourName your.email@example.com"'
    )


def sec_headers() -> dict:
    return {
        "User-Agent": get_sec_user_agent(),
        "Accept-Encoding": "gzip, deflate",
    }
