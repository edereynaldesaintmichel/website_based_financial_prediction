#!/usr/bin/env python3
"""
Process an Athena CSV export of Common Crawl index records into a clean
per-domain manifest for downstream WARC fetching.

Reads the CSV, filters out non-content URLs (robots.txt, .txt files,
privacy/legal pages, etc.), deduplicates by URL across crawls (keeping
the priority crawl), and selects up to 20 pages per domain prioritised
by URL depth (shallow first).

Output: cc_data/manifest.json
  {
    "example.com": [
      {"url": "...", "warc_filename": "...", "offset": "...", "length": "...", "crawl": "..."},
      ...
    ],
    ...
  }

Usage:
  python process_cc_csv.py \\
      --csv ~/Downloads/48051bf0-d201-415b-a57a-df523f03e7d2.csv \\
      [--output cc_data/manifest.json] \\
      [--page-cap 20]
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from urllib.parse import urlparse

from utils import (
    BULLSHIT_PATTERNS,
    CC_CRAWL_PRIORITY,
    PAGE_CAP,
    is_bullshit_page,
    url_depth,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Filtering ────────────────────────────────────────────────────────────────

def should_exclude_url(url: str) -> bool:
    """Return True for URLs that should be filtered out entirely."""
    path = urlparse(url.lower()).path
    if path.endswith(".txt"):
        return True
    if is_bullshit_page(url):
        return True
    return False


# ── Dedup & selection ────────────────────────────────────────────────────────

def _normalize_url(url: str) -> str:
    """Normalise URL for dedup: lowercase host, strip www., strip trailing /."""
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower().removeprefix("www.")
    path = parsed.path.rstrip("/") or "/"
    return f"{parsed.scheme}://{host}{path}"


def pick_best_crawl(records: list[dict]) -> dict:
    """Given duplicate records for the same URL, keep the priority crawl."""
    crawl_rank = {c: i for i, c in enumerate(CC_CRAWL_PRIORITY)}
    records.sort(key=lambda r: crawl_rank.get(r["crawl"], len(CC_CRAWL_PRIORITY)))
    return records[0]


def select_pages(records: list[dict], cap: int = PAGE_CAP) -> list[dict]:
    """
    From all records for one domain:
    1. Deduplicate by normalised URL (keep priority crawl)
    2. Sort by depth (root first), then alphabetically by path
    3. Return top `cap` records.
    """
    by_url: dict[str, list[dict]] = {}
    for r in records:
        norm = _normalize_url(r["url"])
        by_url.setdefault(norm, []).append(r)

    deduped = [pick_best_crawl(recs) for recs in by_url.values()]

    deduped.sort(key=lambda r: (
        url_depth(r["url"]),
        0 if "html" in r.get("mime", "") else 1,
        urlparse(r["url"]).path,
    ))

    return deduped[:cap]


# ── Main ─────────────────────────────────────────────────────────────────────

def process_csv(csv_path: str, output_path: str, page_cap: int) -> None:
    rows_total = 0
    rows_filtered = 0
    by_domain: dict[str, list[dict]] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_total += 1
            url = row["url"]
            if should_exclude_url(url):
                rows_filtered += 1
                continue
            domain = row["domain"]
            by_domain.setdefault(domain, []).append({
                "url":           url,
                "warc_filename": row["warc_filename"],
                "offset":        row["offset"],
                "length":        row["length"],
                "crawl":         row["crawl"],
                "mime":          row["mime"],
            })

    logger.info(
        "CSV: %d rows total, %d filtered, %d remaining across %d domains",
        rows_total, rows_filtered, rows_total - rows_filtered, len(by_domain),
    )

    manifest: dict[str, list[dict]] = {}
    total_pages = 0
    for domain, records in by_domain.items():
        selected = select_pages(records, cap=page_cap)
        # Drop the 'mime' key from the final manifest (not needed downstream)
        for rec in selected:
            rec.pop("mime", None)
        manifest[domain] = selected
        total_pages += len(selected)

    logger.info(
        "Manifest: %d domains, %d total pages (cap=%d/domain)",
        len(manifest), total_pages, page_cap,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Written to %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        default="/Users/eloireynal/Downloads/48051bf0-d201-415b-a57a-df523f03e7d2.csv",
        help="Path to Athena CSV export",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "cc_data" / "manifest.json"),
        help="Output manifest JSON path (default: %(default)s)",
    )
    parser.add_argument(
        "--page-cap",
        type=int,
        default=PAGE_CAP,
        help="Maximum pages per domain (default: %(default)s)",
    )
    args = parser.parse_args()
    process_csv(args.csv, args.output, args.page_cap)


if __name__ == "__main__":
    main()
