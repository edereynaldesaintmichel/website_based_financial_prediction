"""
Compute Annual Growth Rates for SEC-filed US Companies
======================================================
Compares financial metrics between two time periods (e.g. 2018 vs 2025)
by loading per-company JSON files from two directories.  For each company
present in both, computes a growth score based on clipped log-ratios of
shared positive-valued metrics.

Usage:
    python compute_growth_rates.py \
        --early-dir output/financials_json_2018 \
        --late-dir  output/financials_json_2025 \
        --output growth_scores.json \
        --growth-rates-output annual_growth_rates.json
"""

import argparse
import json
import math
import os
import re
from datetime import date


LOG_RATIO_CLIP = 3.0   # cap at ~20x growth or ~95% decline
MIN_SHARED_METRICS = 2


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


def pool_all_dates(financials: dict, prefer: str = "latest"
                   ) -> tuple[dict, str]:
    """Pool metrics across all dates in a company's financials.

    Args:
        financials: {date_str: {metric: {value, unit}}}
        prefer: "latest" keeps the last occurrence per metric,
                "earliest" keeps the first occurrence.
    Returns:
        (pooled_record, representative_date)
    """
    date_keys = sorted(
        k for k in financials if re.match(r"\d{4}-\d{2}-\d{2}", k)
    )
    if not date_keys:
        return {}, ""

    record = {}
    metric_date = {}

    if prefer == "latest":
        # Iterate newest-first so first-seen = latest value
        for dk in reversed(date_keys):
            for m, v in financials[dk].items():
                if m not in record:
                    record[m] = v
                    metric_date[m] = dk
    else:
        for dk in date_keys:
            for m, v in financials[dk].items():
                if m not in record:
                    record[m] = v
                    metric_date[m] = dk

    # Representative date: median of actually-used dates
    used_dates = sorted(set(metric_date.values()))
    rep_date = used_dates[len(used_dates) // 2]

    return record, rep_date


def compute_growth(record_early: dict, record_late: dict) -> dict:
    """Compute mean clipped log-ratio across all shared positive-valued metrics."""
    common_metrics = set(record_early.keys()) & set(record_late.keys())

    log_ratios = []
    for metric in sorted(common_metrics):
        val_early = parse_value(record_early[metric].get("value", ""))
        val_late = parse_value(record_late[metric].get("value", ""))

        if val_early is None or val_late is None:
            continue
        if val_early <= 0 or val_late <= 0:
            continue

        lr = math.log(val_late / val_early)
        lr_clipped = max(-LOG_RATIO_CLIP, min(LOG_RATIO_CLIP, lr))
        log_ratios.append(lr_clipped)

    if len(log_ratios) < MIN_SHARED_METRICS:
        return {"score": None, "fields_used": len(log_ratios)}

    score = sum(log_ratios) / len(log_ratios)
    return {"score": round(score, 6), "fields_used": len(log_ratios)}


def load_company(filepath: str) -> dict | None:
    """Load a company JSON and return parsed structure, or None on error."""
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute annual growth rates for SEC-filed US companies"
    )
    parser.add_argument("--early-dir", type=str, required=True,
                        help="Directory with baseline financials (e.g. 2018)")
    parser.add_argument("--late-dir", type=str, required=True,
                        help="Directory with later financials (e.g. 2025)")
    parser.add_argument("--output", type=str, default="growth_scores.json",
                        help="Path to detailed growth scores JSON list")
    parser.add_argument("--growth-rates-output", type=str,
                        default="annual_growth_rates.json",
                        help="Path to flat {cik: annual_growth_rate} JSON")
    return parser.parse_args()


def main():
    args = parse_args()

    early_files = {f for f in os.listdir(args.early_dir) if f.endswith(".json")}
    late_files = {f for f in os.listdir(args.late_dir) if f.endswith(".json")}
    common_files = sorted(early_files & late_files)

    print(f"Early dir:  {len(early_files)} files in {args.early_dir}")
    print(f"Late dir:   {len(late_files)} files in {args.late_dir}")
    print(f"In common:  {len(common_files)} companies")

    results = []
    skipped_no_dates = 0
    skipped_few_metrics = 0

    for fname in common_files:
        early_data = load_company(os.path.join(args.early_dir, fname))
        late_data = load_company(os.path.join(args.late_dir, fname))
        if early_data is None or late_data is None:
            continue

        identifier = early_data.get("identifier", {})
        cik = identifier.get("cik", "")
        ticker = identifier.get("ticker", "")
        entity_name = identifier.get("entity_name", "")

        early_fins = early_data.get("financials", {})
        late_fins = late_data.get("financials", {})

        # Pool all dates within each period
        early_rec, early_date = pool_all_dates(early_fins, prefer="latest")
        late_rec, late_date = pool_all_dates(late_fins, prefer="latest")

        if not early_date or not late_date:
            skipped_no_dates += 1
            continue

        years = date_diff_years(early_date, late_date)
        if years < 1.0:
            skipped_no_dates += 1
            continue

        growth = compute_growth(early_rec, late_rec)
        if growth["score"] is None:
            skipped_few_metrics += 1
            continue

        annual_score = round(growth["score"] / years, 6)

        results.append({
            "cik": cik,
            "ticker": ticker,
            "entity_name": entity_name,
            "date_early": early_date,
            "date_late": late_date,
            "years": round(years, 2),
            "growth_score": growth["score"],
            "annual_growth_rate": annual_score,
            "fields_used": growth["fields_used"],
        })

    # Sort by annual growth rate descending
    results.sort(
        key=lambda x: x["annual_growth_rate"]
        if x["annual_growth_rate"] is not None else float("-inf"),
        reverse=True,
    )

    # Save detailed results
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    # Save flat CIK -> rate mapping
    growth_rates = {
        r["cik"]: r["annual_growth_rate"]
        for r in results if r["annual_growth_rate"] is not None
    }
    with open(args.growth_rates_output, "w", encoding="utf-8") as fh:
        json.dump(growth_rates, fh, indent=2, ensure_ascii=False)

    # Summary statistics
    annual_scores = [r["annual_growth_rate"] for r in results
                     if r["annual_growth_rate"] is not None]

    print(f"\nProcessed: {len(results)} companies with valid growth scores")
    print(f"Skipped:   {skipped_no_dates} (no dates / too close), "
          f"{skipped_few_metrics} (< {MIN_SHARED_METRICS} shared metrics)")

    if annual_scores:
        s = sorted(annual_scores)
        n = len(s)
        print(f"\nAnnual growth rate summary ({n} companies):")
        print(f"  Mean:   {sum(s) / n:+.4f}")
        print(f"  Median: {s[n // 2]:+.4f}")
        print(f"  Min:    {s[0]:+.4f}")
        print(f"  Max:    {s[-1]:+.4f}")
        print(f"  P10:    {s[int(n * 0.1)]:+.4f}")
        print(f"  P90:    {s[int(n * 0.9)]:+.4f}")

        positive = sum(1 for v in s if v > 0)
        print(f"  Grew:   {positive} ({100 * positive / n:.1f}%)")
        print(f"  Shrank: {n - positive} ({100 * (n - positive) / n:.1f}%)")

    print(f"\nSaved detailed scores to {args.output}")
    print(f"Saved flat growth rates to {args.growth_rates_output}")


if __name__ == "__main__":
    main()
