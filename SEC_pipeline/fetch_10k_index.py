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

import argparse
import csv
import os
import re
import time

import requests
from tqdm import tqdm

from config import (
    EDGAR_FULL_INDEX, FULL_INDEX_URLS, INDEX_DIR, QUARTERS, YEAR,
    index_csv_for_year, sec_headers, REQUEST_DELAY,
)


#   company name  form_type  cik         date         filename
# e.g. "10x Genomics, Inc.   10-K   1770787   2021-02-26   edgar/data/..."
# Anchored on: filename always starts with "edgar/", date is YYYY-MM-DD,
# CIK is a run of digits, form_type is whatever is between company name and CIK.
ROW_RE = re.compile(
    r"^(?P<company>.+?)\s{2,}"         # company name, greedy-lazy, followed by 2+ spaces
    r"(?P<form>\S+(?:\s\S+)*?)\s{2,}"  # form type (may contain single spaces, e.g. "NT 10-K")
    r"(?P<cik>\d+)\s+"                   # CIK
    r"(?P<date>\d{4}-\d{2}-\d{2})\s+"  # filing date
    r"(?P<filename>edgar/\S+)\s*$"     # EDGAR path
)


def fetch_company_idx(url: str, headers: dict) -> list[dict]:
    """Download and parse a single company.idx file from EDGAR.

    EDGAR's full-index is nominally fixed-width but column positions drift
    across quarters when any field exceeds its default width. We parse each
    data row with a regex anchored on the date and ``edgar/...`` filename,
    which are the only two fields with known shapes.
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
        m = ROW_RE.match(line)
        if not m:
            continue
        entries.append({
            "cik": m.group("cik"),
            "company_name": m.group("company").strip(),
            "form_type": m.group("form").strip(),
            "date_filed": m.group("date"),
            "filename": m.group("filename").strip(),
        })

    return entries


def filter_10k(entries: list[dict]) -> list[dict]:
    """Keep only 10-K filings (exact match, not 10-K/A amendments)."""
    return [e for e in entries if e["form_type"] == "10-K"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch SEC EDGAR 10-K filing index (optionally for a single year)."
    )
    parser.add_argument(
        "--year", type=int, default=None,
        help="Restrict to filings filed in this year. Default: use config range.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    headers = sec_headers()
    os.makedirs(INDEX_DIR, exist_ok=True)

    if args.year:
        urls = [
            f"{EDGAR_FULL_INDEX}/{args.year}/QTR{q}/company.idx"
            for q in QUARTERS
        ]
        output_path = index_csv_for_year(args.year)
        year_label = str(args.year)
    else:
        urls = FULL_INDEX_URLS
        output_path = INDEX_DIR / f"10k_filings_{YEAR}.csv"
        year_label = str(YEAR)

    all_filings = []

    print(f"Fetching EDGAR full index for {year_label} ...")
    for url in tqdm(urls, desc="Quarters"):
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
