"""
Check Missing Companies via Companies House API
=================================================
Takes a missing_companies.json (list of company IDs) and queries the
Companies House REST API to determine what happened to each one.

You need a free API key from https://developer.company-information.service.gov.uk/

Usage:
    python check_missing_companies.py \
        --api-key YOUR_API_KEY \
        --input missing_companies.json \
        --output-dir ./company_status_results
"""

import argparse
import json
import logging
import os
import time
from collections import Counter

import requests
from tqdm import tqdm

API_BASE = "https://api.company-information.service.gov.uk"
# Companies House allows 600 requests per 5 minutes = 2 req/s
RATE_LIMIT_DELAY = 0.5  # seconds between requests (safe margin)
BATCH_SAVE_INTERVAL = 50  # save progress every N companies


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return logging.getLogger(__name__)


def get_company_info(company_id: str, api_key: str) -> dict:
    """Query Companies House API for a single company."""
    url = f"{API_BASE}/company/{company_id}"
    response = requests.get(url, auth=(api_key, ""), timeout=30)

    if response.status_code == 200:
        data = response.json()
        return {
            "company_id": company_id,
            "company_name": data.get("company_name"),
            "company_status": data.get("company_status"),
            "company_status_detail": data.get("company_status_detail"),
            "type": data.get("type"),
            "date_of_creation": data.get("date_of_creation"),
            "date_of_cessation": data.get("date_of_cessation"),
            "has_been_liquidated": data.get("has_been_liquidated"),
            "has_charges": data.get("has_charges"),
            "http_status": 200,
        }
    elif response.status_code == 404:
        return {
            "company_id": company_id,
            "company_status": "not_found",
            "http_status": 404,
        }
    else:
        return {
            "company_id": company_id,
            "company_status": "api_error",
            "http_status": response.status_code,
            "error": response.text[:200],
        }


def save_progress(results: list[dict], output_path: str):
    tmp = output_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check what happened to missing companies via Companies House API"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="Companies House API key",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to missing_companies.json (list of company IDs)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./company_status_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous partial run (skips already-checked IDs)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load missing company IDs
    with open(args.input, "r") as fh:
        company_ids = json.load(fh)
    logger.info("Loaded %d company IDs to check", len(company_ids))

    # Resume support
    results_path = os.path.join(args.output_dir, "company_statuses.json")
    results: list[dict] = []
    already_checked: set[str] = set()

    if args.resume and os.path.exists(results_path):
        with open(results_path, "r") as fh:
            results = json.load(fh)
        already_checked = {r["company_id"] for r in results}
        logger.info("Resuming — %d already checked, %d remaining",
                     len(already_checked), len(company_ids) - len(already_checked))

    remaining = [cid for cid in company_ids if cid not in already_checked]

    for i, company_id in enumerate(tqdm(remaining, desc="Checking companies")):
        try:
            info = get_company_info(company_id, args.api_key)
            results.append(info)

            if info["http_status"] == 429:
                logger.warning("Rate limited — sleeping 60s")
                time.sleep(60)

        except Exception as exc:
            logger.warning("Error checking %s: %s", company_id, exc)
            results.append({
                "company_id": company_id,
                "company_status": "request_error",
                "error": str(exc),
            })

        # Save progress periodically
        if (i + 1) % BATCH_SAVE_INTERVAL == 0:
            save_progress(results, results_path)

        time.sleep(RATE_LIMIT_DELAY)

    # Final save
    save_progress(results, results_path)

    # Summary
    status_counts = Counter(r.get("company_status") for r in results)
    summary_path = os.path.join(args.output_dir, "status_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(dict(status_counts.most_common()), fh, indent=2)

    logger.info("")
    logger.info("=" * 50)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 50)
    for status, count in status_counts.most_common():
        pct = 100 * count / len(results)
        logger.info("  %-25s %5d  (%5.1f%%)", status, count, pct)
    logger.info("=" * 50)
    logger.info("Total checked: %d", len(results))
    logger.info("")
    logger.info("Full results : %s", results_path)
    logger.info("Summary      : %s", summary_path)


if __name__ == "__main__":
    main()
