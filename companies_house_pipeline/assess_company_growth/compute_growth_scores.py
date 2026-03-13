"""
Compute Company Growth Scores (2018 → 2025)
============================================
For each company present in both the 2018 and 2025 parsed XBRL directories,
computes a weighted growth score based on log-ratios of common financial fields.
Outputs both total and annualized scores.

Usage:
    python compute_growth_scores.py \
        --dir-2018 /Users/eloireynal/Downloads/json_50plus_employees \
        --dir-2025 ../../xbrl_parsing_pipeline/output/json_50plus_employees \
        --fields common_fields.json \
        --output growth_scores.json
"""

import argparse
import json
import math
import os
import re
from datetime import date


LOG_RATIO_CLIP = 3.0  # cap at ~20x growth or ~95% decline


def extract_company_files(directory: str) -> dict[str, list[str]]:
    """Map company IDs to their JSON filenames, sorted most-recent first."""
    companies: dict[str, list[str]] = {}
    for fname in os.listdir(directory):
        if not fname.endswith("_parsed.json"):
            continue
        parts = fname.split("_")
        if len(parts) >= 4:
            company_id = parts[2]
            companies.setdefault(company_id, []).append(fname)
    for fnames in companies.values():
        fnames.sort(reverse=True)
    return companies


def get_most_recent_record(filepath: str) -> tuple[str | None, dict | None]:
    """Return (date_key, fields_dict) for the most recent filing."""
    with open(filepath, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    date_keys = sorted(
        [k for k in data if re.match(r"\d{4}-\d{2}-\d{2}", k)],
        reverse=True,
    )
    if not date_keys:
        return None, None
    return date_keys[0], data[date_keys[0]]


def parse_value(raw: str) -> float | None:
    """Extract numeric value from '<number>123.0</number>' format."""
    m = re.search(r"<number>([-\d.eE+]+)</number>", str(raw))
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def date_diff_years(d1: str, d2: str) -> float:
    """Return the difference in years between two YYYY-MM-DD date strings."""
    a = date.fromisoformat(d1)
    b = date.fromisoformat(d2)
    return abs((b - a).days) / 365.25


def compute_growth(record_2018: dict, record_2025: dict,
                   field_weights: dict[str, float]) -> dict:
    """Compute weighted log-ratio growth score for a single company."""
    weighted_sum = 0.0
    abs_weight_sum = 0.0
    fields_used = 0

    for field, weight in field_weights.items():
        if field not in record_2018 or field not in record_2025:
            continue

        val_2018 = parse_value(record_2018[field].get("value", ""))
        val_2025 = parse_value(record_2025[field].get("value", ""))

        if val_2018 is None or val_2025 is None:
            continue
        if val_2018 <= 0 or val_2025 <= 0:
            continue

        log_ratio = math.log(val_2025 / val_2018)
        log_ratio_clipped = max(-LOG_RATIO_CLIP, min(LOG_RATIO_CLIP, log_ratio))

        weighted_sum += weight * log_ratio_clipped
        abs_weight_sum += abs(weight)
        fields_used += 1

    if abs_weight_sum == 0:
        return {"score": None, "fields_used": 0}

    score = weighted_sum / abs_weight_sum
    return {"score": round(score, 4), "fields_used": fields_used}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute company growth scores from 2018 to 2025"
    )
    parser.add_argument("--dir-2018", type=str, required=True)
    parser.add_argument("--dir-2025", type=str, required=True)
    parser.add_argument("--fields", type=str, default="common_fields.json",
                        help="Path to common_fields.json with field weights")
    parser.add_argument("--output", type=str, default="growth_scores.json")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.fields, "r") as fh:
        field_weights = json.load(fh)
    field_weights = {k: v for k, v in field_weights.items() if v != 0.0}

    print(f"Loaded {len(field_weights)} fields with non-zero weights")

    companies_2018 = extract_company_files(args.dir_2018)
    companies_2025 = extract_company_files(args.dir_2025)
    common_ids = sorted(set(companies_2018) & set(companies_2025))
    print(f"Matched companies: {len(common_ids)}")

    results = []
    annual_scores = []

    for company_id in common_ids:
        file_2018 = os.path.join(args.dir_2018, companies_2018[company_id][0])
        file_2025 = os.path.join(args.dir_2025, companies_2025[company_id][0])

        date_2018, record_2018 = get_most_recent_record(file_2018)
        date_2025, record_2025 = get_most_recent_record(file_2025)

        if record_2018 is None or record_2025 is None:
            continue

        growth = compute_growth(record_2018, record_2025, field_weights)
        if growth["score"] is None:
            continue

        years = date_diff_years(date_2018, date_2025)
        annual_score = round(growth["score"] / years, 4) if years > 0 else None

        results.append({
            "company_id": company_id,
            "date_2018": date_2018,
            "date_2025": date_2025,
            "years": round(years, 2),
            "growth_score": growth["score"],
            "annual_growth_score": annual_score,
            "fields_used": growth["fields_used"],
        })
        if annual_score is not None:
            annual_scores.append(annual_score)

    results.sort(key=lambda x: x["annual_growth_score"] if x["annual_growth_score"] is not None else float("-inf"), reverse=True)

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    if annual_scores:
        s = sorted(annual_scores)
        n = len(s)
        print(f"\nAnnual growth score summary ({n} companies):")
        print(f"  Mean:   {sum(s) / n:+.4f}")
        print(f"  Median: {s[n // 2]:+.4f}")
        print(f"  Min:    {s[0]:+.4f}")
        print(f"  Max:    {s[-1]:+.4f}")
        print(f"  P10:    {s[int(n * 0.1)]:+.4f}")
        print(f"  P90:    {s[int(n * 0.9)]:+.4f}")

        positive = sum(1 for v in s if v > 0)
        print(f"  Grew:   {positive} ({100 * positive / n:.1f}%)")
        print(f"  Shrank: {n - positive} ({100 * (n - positive) / n:.1f}%)")

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
