"""
SEC 10-K Filing Index Fetcher (2018)
=====================================
Downloads EDGAR quarterly full-index files for 2018, filters for 10-K
annual report filings, and produces a consolidated CSV manifest.

Output: output/index/10k_filings_2018.csv
Columns: cik, company_name, form_type, date_filed, filename

Usage:
    pip install requests tqdm
    python fetch_10k_index.py
"""

import csv
import os
import re
import time

import requests
from tqdm import tqdm

from config import (
    FULL_INDEX_URLS, INDEX_DIR, YEAR, sec_headers, REQUEST_DELAY,
)


def fetch_company_idx(url: str, headers: dict) -> list[dict]:
    """Download and parse a single company.idx file from EDGAR.

    The format is fixed-width with a header section (lines starting with
    dashes or field names) followed by data rows:
        Company Name | Form Type | CIK | Date Filed | Filename
    """
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()

    lines = resp.text.splitlines()

    # Find the data start: line of dashes separates header from data
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith("---"):
            data_start = i + 1
            break

    entries = []
    for line in lines[data_start:]:
        if not line.strip():
            continue
        # Fixed-width columns: Company Name (62), Form Type (12), CIK (12),
        # Date Filed (12), Filename (rest)
        company_name = line[:62].strip()
        form_type = line[62:74].strip()
        cik = line[74:86].strip()
        date_filed = line[86:98].strip()
        filename = line[98:].strip()

        entries.append({
            "cik": cik,
            "company_name": company_name,
            "form_type": form_type,
            "date_filed": date_filed,
            "filename": filename,
        })

    return entries


def filter_10k(entries: list[dict]) -> list[dict]:
    """Keep only 10-K filings (exact match, not 10-K/A amendments)."""
    return [e for e in entries if e["form_type"] == "10-K"]


def main():
    headers = sec_headers()
    os.makedirs(INDEX_DIR, exist_ok=True)

    all_filings = []

    print(f"Fetching EDGAR full index for {YEAR} ...")
    for url in tqdm(FULL_INDEX_URLS, desc="Quarters"):
        qtr = re.search(r"QTR(\d)", url).group(1)
        print(f"\n  Quarter {qtr}: {url}")

        entries = fetch_company_idx(url, headers)
        ten_k = filter_10k(entries)
        print(f"    Total filings: {len(entries):,}  |  10-K filings: {len(ten_k):,}")
        all_filings.extend(ten_k)

        time.sleep(REQUEST_DELAY)

    # Deduplicate by CIK + date (a company might refile)
    seen = set()
    unique = []
    for f in all_filings:
        key = (f["cik"], f["date_filed"])
        if key not in seen:
            seen.add(key)
            unique.append(f)

    output_path = INDEX_DIR / f"10k_filings_{YEAR}.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["cik", "company_name", "form_type", "date_filed", "filename"])
        writer.writeheader()
        writer.writerows(unique)

    print(f"\n{'=' * 60}")
    print(f"10-K filings found: {len(unique):,}")
    print(f"Saved to: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
