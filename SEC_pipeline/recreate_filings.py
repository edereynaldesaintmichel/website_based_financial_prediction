"""
Recreate SEC 10-K filings as clean Markdown.

Merges two data sources:
1. Sanitized HTML (tables removed) -> converted to Markdown via markdownify
2. Parsed financials JSON (from parse_financials.py) -> appended as KV-format tables

Output: output/10k_markdown/<CIK>_<date>.md

Usage:
    python recreate_filings.py [--max N] [--resume]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from markdownify import markdownify as md
from tqdm import tqdm

from config import SANITIZED_HTML_DIR, FINANCIALS_DIR, LLM_SANITIZED_DIR


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


# ──────────────────────────────────────────────────────────────
# HTML -> Markdown
# ──────────────────────────────────────────────────────────────

def html_to_markdown(html: str) -> str:
    """Convert sanitized HTML to clean Markdown."""
    text = md(html, heading_style="ATX", strip=["img"])
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ──────────────────────────────────────────────────────────────
# Financials -> KV Markdown
# ──────────────────────────────────────────────────────────────

def financials_to_table_markdown(financials: dict) -> str:
    """Convert a financials dict (date -> {metric: {value, unit}}) to a Markdown table.

    Picks the most recent date and the date exactly 1 year prior.
    Most recent year comes first (left column).

    Output format:
        # Financial Data

        | Item | 2018-12-31 | 2017-12-31 |
        |------|-----------|-----------|
        | Revenue | 234567 USD | 123456 USD |
        | Net Income | 89000 USD | 78900 USD |
    """
    if not financials:
        return ""

    dates = sorted(financials.keys())
    if not dates:
        return ""

    latest = dates[-1]

    # Find the date exactly 1 year before the latest
    try:
        from datetime import datetime
        latest_dt = datetime.strptime(latest, "%Y-%m-%d")
        prior_dt = latest_dt.replace(year=latest_dt.year - 1)
        prior = prior_dt.strftime("%Y-%m-%d")
    except ValueError:
        prior = None

    # Only include prior if it exists in the data
    if prior and prior in financials:
        selected = [latest, prior]
    else:
        selected = [latest]

    # Collect all metric names across selected dates
    seen = set()
    metric_names = []
    for date in selected:
        for name in sorted(financials[date].keys()):
            if name not in seen:
                seen.add(name)
                metric_names.append(name)

    if not metric_names:
        return ""

    # Header
    header = "| Item | " + " | ".join(selected) + " |"
    separator = "|------|" + "|".join("---" for _ in selected) + "|"

    rows = []
    for name in metric_names:
        cells = []
        for date in selected:
            metric_data = financials[date].get(name)
            if metric_data:
                value = metric_data.get("value", "")
                unit = metric_data.get("unit", "")
                cells.append(f"{value} {unit}".strip())
            else:
                cells.append("")
        rows.append(f"| {name} | " + " | ".join(cells) + " |")

    lines = ["# Financial Data", "", header, separator] + rows
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Load financials index (CIK -> financials data)
# ──────────────────────────────────────────────────────────────

def load_financials_by_cik(financials_dir: str) -> dict:
    """Load all financials JSONs and index them by CIK.

    Returns: { cik_str: { "identifier": {...}, "financials": {...}, ... } }
    """
    index = {}
    financials_path = Path(financials_dir)

    if not financials_path.exists():
        print(f"Warning: financials directory not found: {financials_dir}")
        return index

    for json_path in financials_path.glob("*.json"):
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            cik = data.get("identifier", {}).get("cik")
            if cik:
                index[str(cik)] = data
        except (json.JSONDecodeError, KeyError):
            continue

    return index


# ──────────────────────────────────────────────────────────────
# Merge & save
# ──────────────────────────────────────────────────────────────

def recreate_filing(html_path: str, financials: dict | None) -> str:
    """Produce a single Markdown filing from sanitized HTML + financials."""
    html = read_file(html_path)
    markdown = html_to_markdown(html)

    if financials:
        kv_section = financials_to_table_markdown(financials.get("financials", {}))
        if kv_section:
            markdown = markdown + "\n\n---\n\n" + kv_section

    return markdown


def parse_cik_from_filename(filename: str) -> str | None:
    """Extract CIK from filenames like '1000228_2018-02-21.html'."""
    match = re.match(r"^(\d+)_", filename)
    return match.group(1) if match else None


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Recreate 10-K filings as Markdown (narrative + financial KV tables)"
    )
    parser.add_argument(
        "--html-dir", type=str, default=str(SANITIZED_HTML_DIR),
        help=f"Sanitized HTML input directory (default: {SANITIZED_HTML_DIR})",
    )
    parser.add_argument(
        "--financials-dir", type=str, default=str(FINANCIALS_DIR),
        help=f"Financials JSON directory (default: {FINANCIALS_DIR})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(LLM_SANITIZED_DIR),
        help=f"Markdown output directory (default: {LLM_SANITIZED_DIR})",
    )
    parser.add_argument(
        "--max", type=int, default=0,
        help="Maximum number of filings to process (0 = all)",
    )
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="Skip already-processed files",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    html_dir = Path(args.html_dir)
    output_dir = Path(args.output_dir)

    if not html_dir.exists():
        print(f"Error: HTML directory not found: {html_dir}")
        print("Run sanitize_html.py first.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Load all financials, indexed by CIK
    print("Loading financials JSON files...")
    fin_index = load_financials_by_cik(args.financials_dir)
    print(f"  Loaded financials for {len(fin_index):,} companies")

    # Collect HTML files
    html_files = sorted(html_dir.glob("*.html"))
    if args.max > 0:
        html_files = html_files[:args.max]

    print(f"Processing {len(html_files):,} sanitized HTML files")
    print(f"Output: {output_dir}\n")

    processed = 0
    skipped = 0
    matched = 0

    for html_path in tqdm(html_files, desc="Recreating filings"):
        out_name = html_path.stem + ".md"
        out_path = output_dir / out_name

        if args.resume and out_path.exists():
            skipped += 1
            continue

        cik = parse_cik_from_filename(html_path.name)
        financials = fin_index.get(cik) if cik else None
        if financials:
            matched += 1

        result = recreate_filing(str(html_path), financials)

        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(result)

        processed += 1

    print(f"\n{'=' * 60}")
    print(f"Processed: {processed:,}")
    if skipped:
        print(f"Skipped (already exists): {skipped:,}")
    print(f"Matched with financials: {matched:,}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
