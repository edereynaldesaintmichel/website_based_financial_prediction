"""
Local Mac Script: UK Companies House iXBRL Parser — No Survivorship Bias
========================================================================
Same as xbrl_parser.py, but instead of filtering by employee count,
it only parses/saves filings for companies whose company ID appears
in a reference directory of JSON files (e.g. the 2018 50+ employee set).

This ensures that even companies that shrank, dissolved, or were acquired
between the reference year and now are included in the output.

The company ID is extracted from filenames like:
    Prod224_XXXX_COMPANYID_DATE.html
    Prod224_XXXX_COMPANYID_DATE_parsed.json

Usage:
    source env/bin/activate
    python xbrl_parser_no_survivorship_bias.py \
        --reference-dir /Users/eloireynal/Downloads/json_50plus_employees/ \
        [--workers N] [--output-dir PATH] [--start-from N]
"""

import argparse
import json
import logging
import os
import re
import shutil
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import requests
from lxml import etree
from tqdm import tqdm

try:
    from ixbrl_parse.ixbrl import parse as ixbrl_parse
except ImportError:
    raise SystemExit("❌ Missing dependency: pip install ixbrl-parse lxml requests tqdm")


# ============================================================================
# Configuration
# ============================================================================

ZIP_URLS = [
    "https://download.companieshouse.gov.uk/Accounts_Monthly_Data-January2025.zip",
    "https://download.companieshouse.gov.uk/Accounts_Monthly_Data-February2025.zip",
    "https://download.companieshouse.gov.uk/Accounts_Monthly_Data-March2025.zip",
    "https://download.companieshouse.gov.uk/Accounts_Monthly_Data-April2025.zip",
    "https://download.companieshouse.gov.uk/Accounts_Monthly_Data-May2025.zip",
    "https://download.companieshouse.gov.uk/Accounts_Monthly_Data-June2025.zip",
    "https://download.companieshouse.gov.uk/Accounts_Monthly_Data-July2025.zip",
    "https://download.companieshouse.gov.uk/Accounts_Monthly_Data-August2025.zip",
    "https://download.companieshouse.gov.uk/Accounts_Monthly_Data-September2025.zip",
    "https://download.companieshouse.gov.uk/Accounts_Monthly_Data-October2025.zip",
    "https://download.companieshouse.gov.uk/Accounts_Monthly_Data-November2025.zip",
    "https://download.companieshouse.gov.uk/Accounts_Monthly_Data-December2025.zip",
]

DEFAULT_OUTPUT_DIR = "/Users/eloireynal/Documents/My projects/website_based_financial_prediction/companies_house_pipeline/xbrl_parsing_pipeline/output"
DEFAULT_WORKERS = max(1, (os.cpu_count() or 4) - 1)
DOWNLOAD_CHUNK_SIZE = 1 * 1024 * 1024  # 1 MB

# Regex to extract company ID (3rd underscore-delimited field) from filenames
# e.g. "Prod224_0050_00222839_20170928.html" -> "00222839"
COMPANY_ID_RE = re.compile(r"^[^_]+_[^_]+_([^_]+)_")


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class RunStats:
    """Thread-safe accumulator for processing statistics."""
    processed: int = 0
    errors: int = 0
    matched: int = 0
    skipped_not_in_ref: int = 0
    zips_ok: int = 0
    zips_failed: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, processed=0, errors=0, matched=0, skipped_not_in_ref=0):
        with self._lock:
            self.processed += processed
            self.errors += errors
            self.matched += matched
            self.skipped_not_in_ref += skipped_not_in_ref


# ============================================================================
# Reference set loading
# ============================================================================

def load_reference_company_ids(reference_dir: str, logger: logging.Logger) -> set[str]:
    """Extract the set of company IDs from JSON filenames in reference_dir."""
    ref_path = Path(reference_dir)
    if not ref_path.is_dir():
        raise SystemExit(f"❌ Reference directory does not exist: {reference_dir}")

    company_ids = set()
    for f in ref_path.iterdir():
        if not f.name.endswith(".json"):
            continue
        m = COMPANY_ID_RE.match(f.name)
        if m:
            company_ids.add(m.group(1))

    logger.info("📋 Loaded %d unique company IDs from reference directory", len(company_ids))
    return company_ids


def extract_company_id(filename: str) -> str | None:
    """Extract company ID from a filename like Prod224_XXXX_COMPANYID_DATE.html."""
    m = COMPANY_ID_RE.match(filename)
    return m.group(1) if m else None


# ============================================================================
# Logging setup
# ============================================================================

def setup_logging(output_dir: str) -> logging.Logger:
    log_path = os.path.join(output_dir, "run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


# ============================================================================
# Download
# ============================================================================

def download_file(url: str, dest_path: str, logger: logging.Logger) -> bool:
    try:
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            basename = os.path.basename(url)
            with (
                open(dest_path, "wb") as fh,
                tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=f"⬇  {basename}",
                    leave=False,
                ) as pbar,
            ):
                for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    fh.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as exc:
        logger.error("Download failed for %s: %s", url, exc)
        return False


# ============================================================================
# Extraction
# ============================================================================

def extract_zip(zip_path: str, extract_to: str, logger: logging.Logger) -> bool:
    try:
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
        return True
    except Exception as exc:
        logger.error("Extraction failed for %s: %s", zip_path, exc)
        return False


# ============================================================================
# iXBRL Parsing
# ============================================================================

def parse_ixbrl_file(file_path: str) -> tuple[dict | None, int | None]:
    """Parse a single iXBRL HTML file. Returns (parsed_data, employee_count)."""
    try:
        tree = etree.parse(file_path)
        xbrl = ixbrl_parse(tree)

        parsed_data: dict = {}
        employee_count: int | None = None

        for _, value in xbrl.values.items():
            unit_name = str(value.unit)
            if unit_name not in ("pure", "GBP"):
                continue

            try:
                value_float = float(
                    str(value).replace(f"({unit_name})", "")
                )
            except (ValueError, TypeError):
                continue

            show_unit = unit_name == "GBP"
            value_name = "".join(
                " " + ch if ch.isupper() else ch
                for ch in value.name.localname
            ).strip()

            date: str | None = None
            if value.context.period:
                date = str(value.context.period.end)
            elif value.context.instant:
                date = str(value.context.instant.instant)
            if not date:
                continue

            parsed_data.setdefault(date, {})[value_name] = {
                "value": f"<number>{value_float}</number>",
                "unit": unit_name if show_unit else "",
            }

            name_lower = value_name.lower()
            if any(
                kw in name_lower
                for kw in ("average number employees", "averagenumberemployees", "employees during period")
            ):
                count = int(value_float)
                if employee_count is None or count > employee_count:
                    employee_count = count

        return parsed_data, employee_count

    except Exception:
        return None, None


def process_single_file(
    html_file: Path,
    json_output_dir: str,
    reference_ids: set[str],
    employee_registry: dict,
    registry_lock: threading.Lock,
) -> tuple[int, int, int, int]:
    """
    Parse one file and save JSON if its company ID is in the reference set.

    Returns:
        (processed, errors, matched, skipped_not_in_ref)
    """
    company_id = extract_company_id(html_file.name)
    if company_id is None or company_id not in reference_ids:
        return 1, 0, 0, 1

    parsed_data, employee_count = parse_ixbrl_file(str(html_file))

    if parsed_data is None:
        return 0, 1, 0, 0

    json_filename = html_file.stem + "_parsed.json"
    json_dest = os.path.join(json_output_dir, json_filename)
    with open(json_dest, "w", encoding="utf-8") as fh:
        json.dump(parsed_data, fh, indent=2, default=str, ensure_ascii=False)

    if employee_count is not None:
        with registry_lock:
            employee_registry[html_file.name] = employee_count

    return 1, 0, 1, 0


# ============================================================================
# Ranking persistence
# ============================================================================

def save_ranking(employee_registry: dict, ranking_file: str, logger: logging.Logger):
    sorted_data = dict(
        sorted(employee_registry.items(), key=lambda x: x[1], reverse=True)
    )
    tmp_path = ranking_file + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(sorted_data, fh, indent=2, ensure_ascii=False)
    os.replace(tmp_path, ranking_file)
    logger.info("📊 Ranking saved — %d companies with employee data", len(sorted_data))


# ============================================================================
# Per-zip pipeline
# ============================================================================

def process_zip(
    zip_url: str,
    work_dir: str,
    json_output_dir: str,
    reference_ids: set[str],
    num_workers: int,
    employee_registry: dict,
    registry_lock: threading.Lock,
    stats: RunStats,
    logger: logging.Logger,
):
    zip_name = os.path.basename(zip_url)
    zip_path = os.path.join(work_dir, zip_name)
    extract_dir = os.path.join(work_dir, zip_name.replace(".zip", "_extracted"))

    # 1. Download
    logger.info("⬇  Downloading %s", zip_name)
    if not download_file(zip_url, zip_path, logger):
        logger.warning("⚠️  Skipping %s — download failed", zip_name)
        stats.zips_failed += 1
        return

    # 2. Extract
    logger.info("📦 Extracting %s", zip_name)
    if not extract_zip(zip_path, extract_dir, logger):
        logger.warning("⚠️  Skipping %s — extraction failed", zip_name)
        _safe_remove_file(zip_path, logger)
        stats.zips_failed += 1
        return

    _safe_remove_file(zip_path, logger)
    logger.info("🗑️  Deleted zip %s", zip_name)

    # 3. Parse matching HTML files in parallel
    html_files = list(Path(extract_dir).rglob("*.html"))
    logger.info("🔍 Scanning %d files from %s with %d workers …", len(html_files), zip_name, num_workers)

    zip_processed = zip_errors = zip_matched = zip_skipped = 0

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(
                process_single_file,
                html_file,
                json_output_dir,
                reference_ids,
                employee_registry,
                registry_lock,
            ): html_file
            for html_file in html_files
        }

        with tqdm(total=len(futures), desc=f"📄 {zip_name}", unit="file") as pbar:
            for future in as_completed(futures):
                try:
                    p, e, m, s = future.result()
                except Exception as exc:
                    logger.debug("Unexpected worker error: %s", exc)
                    p, e, m, s = 0, 1, 0, 0

                zip_processed += p
                zip_errors += e
                zip_matched += m
                zip_skipped += s
                pbar.update(1)
                pbar.set_postfix(matched=zip_matched, errors=zip_errors)

    stats.add(processed=zip_processed, errors=zip_errors, matched=zip_matched, skipped_not_in_ref=zip_skipped)
    stats.zips_ok += 1

    # 4. Cleanup
    _safe_remove_dir(extract_dir, logger)
    logger.info("🧹 Cleaned up extracted files for %s", zip_name)

    logger.info(
        "✅ %s complete — %d scanned, %d matched reference set, %d errors",
        zip_name, zip_processed, zip_matched, zip_errors,
    )


# ============================================================================
# Filesystem helpers
# ============================================================================

def _safe_remove_file(path: str, logger: logging.Logger):
    try:
        os.remove(path)
    except OSError as exc:
        logger.warning("Could not delete file %s: %s", path, exc)


def _safe_remove_dir(path: str, logger: logging.Logger):
    try:
        shutil.rmtree(path)
    except OSError as exc:
        logger.warning("Could not delete directory %s: %s", path, exc)


# ============================================================================
# Entry point
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UK Companies House iXBRL parser — no survivorship bias variant. "
                    "Parses only companies present in a reference directory."
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        # required=True,
        default="/Users/eloireynal/Downloads/json_50plus_employees",
        help="Directory containing JSON files from the reference year (e.g. 2018). "
             "Company IDs are extracted from filenames.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel parsing threads (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Root output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        metavar="N",
        help="Skip the first N-1 months and start from month N (1=January)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = os.path.abspath(args.output_dir)
    json_output_dir = os.path.join(output_dir, "json_reference_companies")
    work_dir = os.path.join(output_dir, "_work")
    ranking_file = os.path.join(output_dir, "sorted_by_employees.json")

    os.makedirs(json_output_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    logger = setup_logging(output_dir)

    # Load reference company IDs
    reference_ids = load_reference_company_ids(args.reference_dir, logger)

    # Banner
    logger.info("=" * 65)
    logger.info("🏢  UK Companies House iXBRL Parser — No Survivorship Bias")
    logger.info("📋  Reference companies : %d unique IDs", len(reference_ids))
    logger.info("⚙️   Worker threads      : %d", args.workers)
    logger.info("📂  Output directory    : %s", output_dir)
    logger.info("=" * 65)

    urls_to_process = ZIP_URLS[args.start_from - 1:]
    if args.start_from > 1:
        logger.info("⏩  Skipping first %d month(s), starting from month %d", args.start_from - 1, args.start_from)

    employee_registry: dict = {}
    registry_lock = threading.Lock()
    stats = RunStats()

    start_time = time.perf_counter()

    for i, zip_url in enumerate(urls_to_process, start=args.start_from):
        logger.info("")
        logger.info("── Archive %d / %d ─────────────────────────────────────────", i, len(ZIP_URLS))
        try:
            process_zip(
                zip_url=zip_url,
                work_dir=work_dir,
                json_output_dir=json_output_dir,
                reference_ids=reference_ids,
                num_workers=args.workers,
                employee_registry=employee_registry,
                registry_lock=registry_lock,
                stats=stats,
                logger=logger,
            )
        except Exception as exc:
            logger.exception("❌ Unexpected error processing %s: %s", zip_url, exc)
            stats.zips_failed += 1
            continue

        save_ranking(employee_registry, ranking_file, logger)

    _safe_remove_dir(work_dir, logger)

    # Final summary
    elapsed = time.perf_counter() - start_time
    logger.info("")
    logger.info("=" * 65)
    logger.info("🎉  DONE")
    logger.info("=" * 65)
    logger.info("📁  Archives processed  : %d / %d", stats.zips_ok, len(urls_to_process))
    logger.info("📄  Files scanned       : %s", f"{stats.processed:,}")
    logger.info("✅  Matched reference   : %s", f"{stats.matched:,}")
    logger.info("⏭️   Skipped (not in ref): %s", f"{stats.skipped_not_in_ref:,}")
    logger.info("⚠️   Parse errors        : %s", f"{stats.errors:,}")
    logger.info("⏱️   Total time          : %.1f s (%.1f min)", elapsed, elapsed / 60)
    logger.info("")

    # How many reference companies were found vs missing
    found_ids = set()
    for fname in os.listdir(json_output_dir):
        cid = extract_company_id(fname)
        if cid:
            found_ids.add(cid)
    missing_ids = reference_ids - found_ids
    logger.info("📊  Reference companies found in 2025 data : %d / %d", len(found_ids), len(reference_ids))
    logger.info("❌  Reference companies NOT found           : %d", len(missing_ids))

    # Save missing companies list
    missing_file = os.path.join(output_dir, "missing_companies.json")
    with open(missing_file, "w", encoding="utf-8") as fh:
        json.dump(sorted(missing_ids), fh, indent=2)
    logger.info("📝  Missing company IDs saved to: %s", missing_file)

    logger.info("")
    logger.info("📂  JSON output    : %s", json_output_dir)
    logger.info("📊  Employee rank  : %s", ranking_file)
    logger.info("📝  Run log        : %s", os.path.join(output_dir, "run.log"))
    logger.info("=" * 65)

    save_ranking(employee_registry, ranking_file, logger)


if __name__ == "__main__":
    main()
