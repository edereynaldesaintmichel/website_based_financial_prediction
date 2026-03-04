"""
Local Mac Script: UK Companies House iXBRL Parser (2025)
=========================================================
Downloads each monthly zip, extracts it, parses all iXBRL files in parallel
using a thread pool, then immediately deletes the zip and extracted files to
keep disk usage minimal.

Only companies with >= EMPLOYEE_THRESHOLD employees are kept.
Output: JSON files + a sorted employee ranking JSON.

Usage:
    pip install lxml ixbrl-parse requests tqdm
    python xbrl_parser.py [--workers N] [--threshold N] [--output-dir PATH]
"""

import argparse
import json
import logging
import os
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
    # "https://download.companieshouse.gov.uk/Accounts_Monthly_Data-January2025.zip",
    # "https://download.companieshouse.gov.uk/Accounts_Monthly_Data-February2025.zip",
    # "https://download.companieshouse.gov.uk/Accounts_Monthly_Data-March2025.zip",
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

DEFAULT_EMPLOYEE_THRESHOLD = 50
DEFAULT_OUTPUT_DIR = "./output"
# Use CPU count as base; iXBRL parsing is CPU-bound but lxml releases the GIL
# for I/O, so threads are effective. Leave one core free for the OS.
DEFAULT_WORKERS = max(1, (os.cpu_count() or 4) - 1)
DOWNLOAD_CHUNK_SIZE = 1 * 1024 * 1024  # 1 MB


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class RunStats:
    """Thread-safe accumulator for processing statistics."""
    processed: int = 0
    errors: int = 0
    above_threshold: int = 0
    zips_ok: int = 0
    zips_failed: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, processed=0, errors=0, above_threshold=0):
        with self._lock:
            self.processed += processed
            self.errors += errors
            self.above_threshold += above_threshold


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
    """
    Stream-download a file to dest_path with a progress bar.
    Returns True on success.
    """
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
    """Extract zip to a dedicated subdirectory. Returns True on success."""
    try:
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
        return True
    except Exception as exc:
        logger.error("Extraction failed for %s: %s", zip_path, exc)
        return False


# ============================================================================
# iXBRL Parsing (runs in worker threads)
# ============================================================================

def parse_ixbrl_file(file_path: str) -> tuple[dict | None, int | None]:
    """
    Parse a single iXBRL HTML file.

    Returns:
        (parsed_data, employee_count) or (None, None) on failure.
    """
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

            # Resolve date
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

            # Employee count detection
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
    employee_threshold: int,
    employee_registry: dict,
    registry_lock: threading.Lock,
) -> tuple[int, int, int]:
    """
    Parse one file and save JSON if it qualifies.

    Returns:
        (processed, errors, above_threshold) — each 0 or 1.
    """
    parsed_data, employee_count = parse_ixbrl_file(str(html_file))

    if parsed_data is None:
        return 0, 1, 0

    if employee_count is not None and employee_count >= employee_threshold:
        json_filename = html_file.stem + "_parsed.json"
        json_dest = os.path.join(json_output_dir, json_filename)
        with open(json_dest, "w", encoding="utf-8") as fh:
            json.dump(parsed_data, fh, indent=2, default=str, ensure_ascii=False)

        with registry_lock:
            employee_registry[html_file.name] = employee_count

        return 1, 0, 1

    return 1, 0, 0


# ============================================================================
# Ranking persistence
# ============================================================================

def save_ranking(employee_registry: dict, ranking_file: str, logger: logging.Logger):
    """Write the sorted employee ranking JSON (intermediate + final)."""
    sorted_data = dict(
        sorted(employee_registry.items(), key=lambda x: x[1], reverse=True)
    )
    tmp_path = ranking_file + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(sorted_data, fh, indent=2, ensure_ascii=False)
    os.replace(tmp_path, ranking_file)  # atomic replace
    logger.info("📊 Ranking saved — %d qualifying companies", len(sorted_data))


# ============================================================================
# Per-zip pipeline
# ============================================================================

def process_zip(
    zip_url: str,
    work_dir: str,
    json_output_dir: str,
    employee_threshold: int,
    num_workers: int,
    employee_registry: dict,
    registry_lock: threading.Lock,
    stats: RunStats,
    logger: logging.Logger,
):
    """
    Full pipeline for one zip archive:
      1. Download  →  2. Extract  →  3. Parse (parallel)
      →  4. Delete extracted files  →  5. Delete zip

    The zip and its extracted contents are removed as soon as parsing
    for that month is complete.
    """
    zip_name = os.path.basename(zip_url)
    zip_path = os.path.join(work_dir, zip_name)
    # Each zip gets its own isolated extract directory so parallel zips
    # (if ever enabled) won't conflict.
    extract_dir = os.path.join(work_dir, zip_name.replace(".zip", "_extracted"))

    # ── 1. Download ──────────────────────────────────────────────────────────
    logger.info("⬇  Downloading %s", zip_name)
    if not download_file(zip_url, zip_path, logger):
        logger.warning("⚠️  Skipping %s — download failed", zip_name)
        stats.zips_failed += 1
        return

    # ── 2. Extract ───────────────────────────────────────────────────────────
    logger.info("📦 Extracting %s", zip_name)
    if not extract_zip(zip_path, extract_dir, logger):
        logger.warning("⚠️  Skipping %s — extraction failed", zip_name)
        _safe_remove_file(zip_path, logger)
        stats.zips_failed += 1
        return

    # Delete zip immediately after extraction — no longer needed
    _safe_remove_file(zip_path, logger)
    logger.info("🗑️  Deleted zip %s", zip_name)

    # ── 3. Parse all HTML files in parallel ──────────────────────────────────
    html_files = list(Path(extract_dir).rglob("*.html"))
    logger.info("🔍 Parsing %d files from %s with %d workers …", len(html_files), zip_name, num_workers)

    zip_processed = zip_errors = zip_above = 0

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(
                process_single_file,
                html_file,
                json_output_dir,
                employee_threshold,
                employee_registry,
                registry_lock,
            ): html_file
            for html_file in html_files
        }

        with tqdm(total=len(futures), desc=f"📄 {zip_name}", unit="file") as pbar:
            for future in as_completed(futures):
                try:
                    p, e, a = future.result()
                except Exception as exc:
                    logger.debug("Unexpected worker error: %s", exc)
                    p, e, a = 0, 1, 0

                zip_processed += p
                zip_errors += e
                zip_above += a
                pbar.update(1)
                pbar.set_postfix(qualifying=zip_above, errors=zip_errors)

    stats.add(processed=zip_processed, errors=zip_errors, above_threshold=zip_above)
    stats.zips_ok += 1

    # ── 4. Delete extracted files ─────────────────────────────────────────────
    _safe_remove_dir(extract_dir, logger)
    logger.info("🧹 Cleaned up extracted files for %s", zip_name)

    logger.info(
        "✅ %s complete — %d parsed, %d qualifying (≥%d employees), %d errors",
        zip_name, zip_processed, zip_above, employee_threshold, zip_errors,
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
        description="UK Companies House iXBRL parser — 2025, local multithreaded"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel parsing threads (default: {DEFAULT_WORKERS}, "
             f"= CPU count - 1 = {os.cpu_count() or 4} - 1)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_EMPLOYEE_THRESHOLD,
        help=f"Minimum employee count to keep a filing (default: {DEFAULT_EMPLOYEE_THRESHOLD})",
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
        help="Skip the first N-1 months and start from month N (1=January, useful for resuming)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve paths
    output_dir = os.path.abspath(args.output_dir)
    json_output_dir = os.path.join(output_dir, "json_50plus_employees")
    work_dir = os.path.join(output_dir, "_work")          # temp downloads + extracts
    ranking_file = os.path.join(output_dir, "sorted_by_employees.json")

    os.makedirs(json_output_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    logger = setup_logging(output_dir)

    # ── Banner ────────────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("🏢  UK Companies House iXBRL Parser  —  2025 Data")
    logger.info("👥  Employee threshold : ≥ %d", args.threshold)
    logger.info("⚙️   Worker threads     : %d", args.workers)
    logger.info("📂  Output directory   : %s", output_dir)
    logger.info("=" * 65)

    urls_to_process = ZIP_URLS[args.start_from - 1:]
    if args.start_from > 1:
        logger.info("⏩  Skipping first %d month(s), starting from month %d", args.start_from - 1, args.start_from)

    # Shared state (thread-safe via lock)
    employee_registry: dict = {}
    registry_lock = threading.Lock()
    stats = RunStats()

    start_time = time.perf_counter()

    # Process each zip sequentially (keeps disk usage to ~1 zip at a time),
    # parallelism is applied *within* each zip across individual files.
    for i, zip_url in enumerate(urls_to_process, start=args.start_from):
        logger.info("")
        logger.info("── Archive %d / %d ─────────────────────────────────────────", i, len(ZIP_URLS))
        try:
            process_zip(
                zip_url=zip_url,
                work_dir=work_dir,
                json_output_dir=json_output_dir,
                employee_threshold=args.threshold,
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

        # Persist intermediate ranking after each month (crash-safe)
        save_ranking(employee_registry, ranking_file, logger)

    # ── Clean up work dir (should be empty, but just in case) ─────────────────
    _safe_remove_dir(work_dir, logger)

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - start_time
    logger.info("")
    logger.info("=" * 65)
    logger.info("🎉  DONE")
    logger.info("=" * 65)
    logger.info("📁  Archives processed : %d / %d", stats.zips_ok, len(urls_to_process))
    logger.info("📄  Files parsed       : %s", f"{stats.processed:,}")
    logger.info("👥  Qualifying companies: %s", f"{stats.above_threshold:,}")
    logger.info("⚠️   Parse errors       : %s", f"{stats.errors:,}")
    logger.info("⏱️   Total time         : %.1f s (%.1f min)", elapsed, elapsed / 60)
    logger.info("")
    logger.info("📂  JSON output    : %s", json_output_dir)
    logger.info("📊  Employee rank  : %s", ranking_file)
    logger.info("📝  Run log        : %s", os.path.join(output_dir, "run.log"))
    logger.info("=" * 65)

    # Print top 20
    sorted_reg = sorted(employee_registry.items(), key=lambda x: x[1], reverse=True)
    if sorted_reg:
        logger.info("")
        logger.info("🏆  Top 20 companies by employee count:")
        for rank, (fname, count) in enumerate(sorted_reg[:20], 1):
            logger.info("  %2d. %-55s %s employees", rank, fname, f"{count:,}")

    save_ranking(employee_registry, ranking_file, logger)


if __name__ == "__main__":
    main()
