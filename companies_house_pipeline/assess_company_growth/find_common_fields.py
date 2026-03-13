"""
Find Common Financial Fields Between 2018 and 2025 Company Filings
==================================================================
Matches companies across two parsed-XBRL directories by Companies House ID,
takes the most recent filing from each, and identifies all financial fields
that appear in both periods. Outputs a JSON dict of {field_name: weight}.

Usage:
    python find_common_fields.py \
        --dir-2018 /Users/eloireynal/Downloads/json_50plus_employees \
        --dir-2025 ../../xbrl_parsing_pipeline/output/json_50plus_employees \
        --output common_fields.json
"""

import argparse
import json
import os
import re
from collections import Counter


def extract_company_files(directory: str) -> dict[str, list[str]]:
    """Map company IDs to their JSON filenames in the given directory."""
    companies: dict[str, list[str]] = {}
    for fname in os.listdir(directory):
        if not fname.endswith("_parsed.json"):
            continue
        parts = fname.split("_")
        if len(parts) >= 4:
            company_id = parts[2]
            companies.setdefault(company_id, []).append(fname)
    # Sort filenames descending so the most recent filing comes first
    for fnames in companies.values():
        fnames.sort(reverse=True)
    return companies


def get_most_recent_record(filepath: str) -> dict | None:
    """Load a parsed JSON and return the fields from the most recent filing date."""
    with open(filepath, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    # Find the most recent date key (skip non-date keys like 'identifier')
    date_keys = sorted(
        [k for k in data if re.match(r"\d{4}-\d{2}-\d{2}", k)],
        reverse=True,
    )
    if not date_keys:
        return None
    return data[date_keys[0]]


def extract_field_names(record: dict) -> set[str]:
    """Extract field names from a filing record, keeping only numeric GBP or dimensionless fields."""
    fields = set()
    for field_name, info in record.items():
        if isinstance(info, dict) and "value" in info:
            fields.add(field_name)
    return fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find common financial fields between 2018 and 2025 filings"
    )
    parser.add_argument(
        "--dir-2018",
        type=str,
        required=True,
        help="Directory with 2018 parsed XBRL JSONs",
    )
    parser.add_argument(
        "--dir-2025",
        type=str,
        required=True,
        help="Directory with 2025 parsed XBRL JSONs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="common_fields.json",
        help="Output path for the common fields JSON (default: common_fields.json)",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.0,
        help="Minimum fraction of matched companies that must have the field (0.0-1.0). "
             "Default 0.0 keeps all fields found in at least one matched pair.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Scanning directories...")
    companies_2018 = extract_company_files(args.dir_2018)
    companies_2025 = extract_company_files(args.dir_2025)

    common_ids = set(companies_2018) & set(companies_2025)
    print(f"  2018 companies: {len(companies_2018)}")
    print(f"  2025 companies: {len(companies_2025)}")
    print(f"  Matched companies: {len(common_ids)}")

    if not common_ids:
        print("No matching companies found. Check directory paths and filename patterns.")
        return

    # For each matched company, find fields present in BOTH the 2018 and 2025 record
    field_counter = Counter()  # how many companies have each field in both periods
    n_matched = 0

    for company_id in sorted(common_ids):
        # Most recent file from each period
        file_2018 = os.path.join(args.dir_2018, companies_2018[company_id][0])
        file_2025 = os.path.join(args.dir_2025, companies_2025[company_id][0])

        record_2018 = get_most_recent_record(file_2018)
        record_2025 = get_most_recent_record(file_2025)

        if record_2018 is None or record_2025 is None:
            continue

        fields_2018 = extract_field_names(record_2018)
        fields_2025 = extract_field_names(record_2025)
        common_fields = fields_2018 & fields_2025

        if common_fields:
            n_matched += 1
            for field in common_fields:
                field_counter[field] += 1

    print(f"\n  Companies with valid records in both periods: {n_matched}")
    print(f"  Total unique common fields found: {len(field_counter)}")

    # Apply minimum coverage filter
    min_count = int(args.min_coverage * n_matched) if n_matched > 0 else 0
    filtered_fields = {
        field: 1.0
        for field, count in field_counter.most_common()
        if count >= max(min_count, 1)
    }

    print(f"  Fields after coverage filter (>= {args.min_coverage:.0%}): {len(filtered_fields)}")

    # Show top fields by coverage
    print(f"\nTop 20 fields by coverage:")
    for field, count in field_counter.most_common(20):
        pct = 100 * count / n_matched if n_matched else 0
        print(f"  {field:<60s} {count:5d} ({pct:5.1f}%)")

    # Save output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(filtered_fields, fh, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(filtered_fields)} fields to {args.output}")


if __name__ == "__main__":
    main()
