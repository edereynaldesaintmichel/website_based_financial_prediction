"""
SEC 10-K HTML Document Downloader (2018)
==========================================
Reads the filing index CSV (from fetch_10k_index.py) and downloads
the primary 10-K HTML document for each filing from EDGAR.

Each EDGAR filing has an index page listing all associated documents.
We parse that index to find the primary 10-K HTML document (the largest
HTML file, which is typically the full annual report).

Output: output/10k_html_raw/<CIK>_<filing_date>.html

Usage:
    pip install requests tqdm beautifulsoup4 lxml
    python download_10k_html.py [--max N] [--resume]
"""

import argparse
import csv
import os
import re
import time

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from config import (
    EDGAR_BASE, INDEX_DIR, RAW_HTML_DIR, YEAR,
    sec_headers, REQUEST_DELAY,
)


def load_index(index_path: str) -> list[dict]:
    """Load the 10-K filing index CSV."""
    with open(index_path, "r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def get_filing_index_url(filename: str) -> str:
    """Convert EDGAR filename (e.g. 'edgar/data/1234/000123456-18-001234.txt')
    to the filing index URL."""
    # The .txt file is a redirect/index; the actual index page is at
    # the directory level with -index.htm
    base = filename.rsplit("/", 1)[0]
    accession = filename.rsplit("/", 1)[1].replace(".txt", "")
    return f"{EDGAR_BASE}/Archives/{base}/{accession}-index.htm"


def find_primary_html(filing_index_url: str, headers: dict) -> str | None:
    """Parse a filing's index page and find the primary 10-K HTML document URL.

    Strategy: look for the largest .htm/.html file in the filing documents table,
    which is typically the full annual report text.
    """
    try:
        resp = requests.get(filing_index_url, headers=headers, timeout=30)
        resp.raise_for_status()
    except Exception:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", class_="tableFile")
    if not table:
        return None

    best_url = None
    best_size = 0

    for row in table.find_all("tr")[1:]:  # skip header row
        cells = row.find_all("td")
        if len(cells) < 5:
            continue

        doc_type = cells[3].get_text(strip=True).lower()
        # Look for the primary document (type description contains "10-k" or
        # is listed as the main filing document)
        link = cells[2].find("a")
        if not link:
            continue

        href = link.get("href", "")
        if not re.search(r"\.html?$", href, re.I):
            continue

        # Parse size (e.g., "2345678" bytes or "2.3 MB")
        size_text = cells[4].get_text(strip=True)
        try:
            size = int(size_text.replace(",", ""))
        except ValueError:
            size = 0

        if size > best_size:
            best_size = size
            best_url = href

    if best_url and not best_url.startswith("http"):
        best_url = f"{EDGAR_BASE}{best_url}"

    return best_url


def download_html(url: str, dest_path: str, headers: dict) -> bool:
    """Download a single HTML document."""
    try:
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        with open(dest_path, "w", encoding="utf-8") as fh:
            fh.write(resp.text)
        return True
    except Exception as exc:
        print(f"    Failed: {exc}")
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Download SEC 10-K HTML documents for {YEAR}"
    )
    parser.add_argument(
        "--max", type=int, default=0,
        help="Maximum number of filings to download (0 = all)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip already-downloaded files", default=True
    )
    return parser.parse_args()


def main():
    args = parse_args()
    headers = sec_headers()

    index_path = INDEX_DIR / f"10k_filings_{YEAR}.csv"
    if not index_path.exists():
        print(f"Error: index file not found: {index_path}")
        print("Run fetch_10k_index.py first.")
        return

    filings = load_index(str(index_path))
    print(f"Loaded {len(filings):,} 10-K filings from index")

    os.makedirs(RAW_HTML_DIR, exist_ok=True)

    if args.max > 0:
        filings = filings[:args.max]
        print(f"Limiting to first {args.max} filings")

    downloaded = 0
    skipped = 0
    errors = 0

    for filing in tqdm(filings, desc="Downloading 10-K HTML"):
        cik = filing["cik"]
        date_filed = filing["date_filed"]
        safe_name = f"{cik}_{date_filed}.html"
        dest_path = os.path.join(str(RAW_HTML_DIR), safe_name)

        # Resume support
        if args.resume and os.path.exists(dest_path):
            skipped += 1
            continue

        # Get filing index page URL
        filing_index_url = get_filing_index_url(filing["filename"])

        # Find the primary HTML document
        time.sleep(REQUEST_DELAY)
        html_url = find_primary_html(filing_index_url, headers)

        if not html_url:
            errors += 1
            continue

        # Download the HTML document
        time.sleep(REQUEST_DELAY)
        if download_html(html_url, dest_path, headers):
            downloaded += 1
        else:
            errors += 1

    print(f"\n{'=' * 60}")
    print(f"Downloaded: {downloaded:,}")
    if skipped:
        print(f"Skipped (already exists): {skipped:,}")
    if errors:
        print(f"Errors: {errors:,}")
    print(f"Output: {RAW_HTML_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
