"""
SEC Financial Data Parser (2018)
=================================
Downloads the SEC Financial Statement Data Sets (pre-parsed XBRL)
for each quarter of 2018, then extracts per-company financial metrics
and saves them as individual JSON files.

This is the SEC equivalent of xbrl_parser.py — the SEC conveniently
provides pre-parsed XBRL data so we don't need to parse raw filings.

Data source: https://www.sec.gov/dera/data/financial-statement-data-sets
Files inside each quarterly ZIP:
    sub.txt — Submission metadata (CIK, company name, form type, ...)
    num.txt — Numeric facts (tag, date, value, unit of measure)
    tag.txt — Tag/concept definitions
    pre.txt — Presentation tree info

Requires: output/company_meta.json (from fetch_company_meta.py) for
ticker-based filenames and website URLs.

Output: output/financials_json/<TICKER>.json
Format matches the UK pipeline: identifier block (with website) +
financials keyed by date, then by metric name, with <number> tags.

Usage:
    pip install requests tqdm
    python fetch_company_meta.py   # run first
    python parse_financials.py [--output-dir PATH]
"""

import argparse
import csv
import json
import io
import os
import time
import zipfile
from collections import defaultdict

import requests
from tqdm import tqdm

from config import (
    DERA_BASE, DERA_ZIPS, FINANCIALS_DIR, OUTPUT_DIR, WORK_DIR,
    START_YEAR, END_YEAR, QUARTERS,
    sec_headers, REQUEST_DELAY, DOWNLOAD_CHUNK_SIZE,
)


# ============================================================================
# Download
# ============================================================================

def download_dera_zip(url: str, dest_path: str, headers: dict) -> bool:
    """Stream-download a DERA ZIP file."""
    try:
        with requests.get(url, stream=True, headers=headers, timeout=120) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            basename = os.path.basename(url)
            with (
                open(dest_path, "wb") as fh,
                tqdm(total=total, unit="B", unit_scale=True,
                     desc=f"  {basename}", leave=False) as pbar,
            ):
                for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    fh.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as exc:
        print(f"  Download failed for {url}: {exc}")
        return False


# ============================================================================
# TSV reading from ZIP
# ============================================================================

def read_tsv_from_zip(zip_path: str, filename: str) -> list[dict]:
    """Read a TSV file from inside a ZIP archive, return list of row dicts."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(filename) as f:
            # DERA files are tab-separated, sometimes with encoding quirks
            text = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
            reader = csv.DictReader(text, delimiter="\t")
            return list(reader)


# ============================================================================
# Core parsing
# ============================================================================

def parse_quarter(zip_path: str) -> dict:
    """Parse one quarterly DERA ZIP and return per-CIK financial data.

    Returns:
        {
            cik: {
                "company_name": str,
                "form_type": str,
                "filed": str,
                "metrics": {
                    "2018-12-31": {
                        "Revenue": {"value": "<number>123456</number>", "unit": "USD"},
                        ...
                    }
                }
            }
        }
    """
    # 1. Read submissions — filter to 10-K only
    subs = read_tsv_from_zip(zip_path, "sub.txt")
    ten_k_adsh = {}  # adsh -> submission info
    for row in subs:
        form = row.get("form", "").strip()
        if form != "10-K":
            continue
        adsh = row.get("adsh", "").strip()
        cik = row.get("cik", "").strip()
        name = row.get("name", "").strip()
        filed = row.get("filed", "").strip()
        ten_k_adsh[adsh] = {
            "cik": cik,
            "company_name": name,
            "form_type": form,
            "filed": filed,
        }

    if not ten_k_adsh:
        return {}

    # 2. Read numeric facts for those submissions
    nums = read_tsv_from_zip(zip_path, "num.txt")

    companies = {}  # cik -> { company_name, metrics: {date: {tag: {value, unit}}} }

    for row in nums:
        adsh = row.get("adsh", "").strip()
        if adsh not in ten_k_adsh:
            continue

        sub_info = ten_k_adsh[adsh]
        cik = sub_info["cik"]

        tag = row.get("tag", "").strip()
        value_str = row.get("value", "").strip()
        uom = row.get("uom", "").strip()
        ddate = row.get("ddate", "").strip()  # YYYYMMDD format
        qtrs = row.get("qtrs", "").strip()    # 0 = instant, 1-4 = duration
        coreg = row.get("coreg", "").strip()  # co-registrant (skip if not empty)

        # Skip co-registrant facts (subsidiaries)
        if coreg:
            continue

        # Parse value
        try:
            value_float = float(value_str)
        except (ValueError, TypeError):
            continue

        # Format date as YYYY-MM-DD
        if len(ddate) == 8:
            date_formatted = f"{ddate[:4]}-{ddate[4:6]}-{ddate[6:8]}"
        else:
            continue

        # Convert camelCase tag to readable name
        tag_readable = _camel_to_title(tag)

        # Determine if we show the unit
        show_unit = uom.upper() in ("USD", "USD/SHARES")
        display_unit = uom.upper() if show_unit else ""

        # Format value: integer if whole, else float
        if value_float == int(value_float) and abs(value_float) < 1e15:
            value_formatted = str(int(value_float))
        else:
            value_formatted = f"{value_float:.10f}".rstrip("0").rstrip(".")

        # Store
        if cik not in companies:
            companies[cik] = {
                "company_name": sub_info["company_name"],
                "form_type": sub_info["form_type"],
                "filed": sub_info["filed"],
                "metrics": defaultdict(dict),
            }

        companies[cik]["metrics"][date_formatted][tag_readable] = {
            "value": f"<number>{value_formatted}</number>",
            "unit": display_unit,
        }

    return companies


def _camel_to_title(name: str) -> str:
    """Convert CamelCase XBRL tag to 'Title Case With Spaces'.

    Examples:
        RevenueFromContractWithCustomer -> Revenue From Contract With Customer
        EarningsPerShareBasic -> Earnings Per Share Basic
    """
    import re
    # Insert space before uppercase letters that follow lowercase
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    # Insert space before sequences of uppercase followed by lowercase
    spaced = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", spaced)
    return spaced


# ============================================================================
# Output
# ============================================================================

def load_company_meta() -> dict:
    """Load the CIK → {ticker, company_name, website} mapping.
    Returns empty dict if not found."""
    meta_path = OUTPUT_DIR / "company_meta.json"
    if not meta_path.exists():
        print(f"Warning: {meta_path} not found. Run fetch_company_meta.py first.")
        print("  Falling back to CIK-based filenames, no website info.")
        return {}
    with open(meta_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_company_json(cik: str, data: dict, output_dir: str,
                      meta: dict) -> None:
    """Save a single company's financial data as JSON, named by ticker."""
    company_meta = meta.get(cik, {})
    ticker = company_meta.get("ticker")

    # Filename: ticker-based if available, else CIK-based fallback
    if ticker:
        filename = f"{ticker}.json"
    else:
        safe_name = "".join(
            c if c.isalnum() or c in " -_" else "_"
            for c in data["company_name"]
        ).strip().replace(" ", "_")[:80]
        filename = f"CIK{cik}_{safe_name}.json"

    filepath = os.path.join(output_dir, filename)

    # Build identifier block (matching UK pipeline structure)
    identifier = {
        "entity_name": data["company_name"],
        "cik": cik,
    }
    if ticker:
        identifier["ticker"] = ticker
    website = company_meta.get("website")
    if website:
        identifier["websites"] = [website]

    # Convert defaultdict to regular dict for JSON serialization
    output = {
        "identifier": identifier,
        "form_type": data["form_type"],
        "filed": data["filed"],
        "financials": dict(data["metrics"]),
    }

    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)


# ============================================================================
# Entry point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Parse SEC Financial Statement Data Sets ({START_YEAR}-{END_YEAR})"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(FINANCIALS_DIR),
        help=f"Output directory for per-company JSON files (default: {FINANCIALS_DIR})",
    )
    parser.add_argument(
        "--year", type=int, default=None,
        help=f"Process a single year only (default: all years {START_YEAR}-{END_YEAR})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    headers = sec_headers()

    # Determine which DERA ZIPs to process
    if args.year:
        dera_zips = [f"{DERA_BASE}/{args.year}q{q}.zip" for q in QUARTERS]
        year_label = str(args.year)
    else:
        dera_zips = DERA_ZIPS
        year_label = f"{START_YEAR}-{END_YEAR}"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(WORK_DIR, exist_ok=True)

    print("=" * 60)
    print(f"SEC Financial Statement Data Sets Parser — {year_label}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Aggregate across all quarters (a company may appear in multiple quarters)
    all_companies = {}  # cik -> merged data

    for url in dera_zips:
        basename = os.path.basename(url)
        zip_path = str(WORK_DIR / basename)
        print(f"\nProcessing {basename} ...")

        # Download
        if not download_dera_zip(url, zip_path, headers):
            continue
        time.sleep(REQUEST_DELAY)

        # Parse
        quarter_data = parse_quarter(zip_path)
        print(f"  10-K filings in this quarter: {len(quarter_data):,}")

        # Merge into all_companies
        for cik, data in quarter_data.items():
            if cik not in all_companies:
                all_companies[cik] = data
            else:
                # Merge metrics (different dates)
                for date, metrics in data["metrics"].items():
                    all_companies[cik]["metrics"].setdefault(date, {}).update(metrics)

        # Clean up ZIP
        try:
            os.remove(zip_path)
        except OSError:
            pass

    # Load company metadata (ticker + website)
    meta = load_company_meta()

    # Save per-company JSON files (named by ticker)
    print(f"\nSaving {len(all_companies):,} company files ...")
    for cik, data in tqdm(all_companies.items(), desc="Saving JSON"):
        save_company_json(cik, data, output_dir, meta)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Done! {len(all_companies):,} companies with 10-K filings saved.")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
