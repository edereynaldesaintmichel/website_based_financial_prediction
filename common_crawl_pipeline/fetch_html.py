#!/usr/bin/env python3
"""
Fetch HTML for company websites from Common Crawl WARC records with a
Wayback Machine fallback for domains absent from the CC manifest.

Pipeline
--------
1. Load a pre-built manifest (from process_cc_csv.py) mapping each domain
   to up to 20 WARC coordinates.
2. For each CC hit: range-request the relevant WARC slice → extract HTML.
3. Load company JSONs to discover domains not in the manifest.
4. For each CC miss: query Wayback Machine CDX → fetch replay HTML
   (up to 20 pages per domain, rate-limited to ~0.5 req/s).

Output layout:
  cc_data/raw_html/<domain>/<slug>.html
  cc_data/raw_html/<domain>/manifest.json

Usage:
  python fetch_html.py \\
      [--manifest cc_data/manifest.json] \\
      [--input-dir /path/to/json_50plus_employees] \\
      [--output-dir cc_data/raw_html] \\
      [--concurrency-warc 8] \\
      [--dry-run]
"""

import argparse
import asyncio
import gzip
import io
import json
import logging
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
import boto3
import tldextract
from tqdm.asyncio import tqdm_asyncio
from warcio.archiveiterator import ArchiveIterator

from utils import (
    PAGE_CAP,
    is_bullshit_page,
    url_depth,
    url_to_slug,
)

# ── Constants ─────────────────────────────────────────────────────────────────

CC_S3_BUCKET = "commoncrawl"

_s3 = boto3.client("s3", region_name="us-east-1")

WB_CDX_URL   = "https://web.archive.org/cdx/search/cdx"
WB_BASE_URL  = "https://web.archive.org/web"
WB_DATE_FROM = "20170101"
WB_DATE_TO   = "20181231"
WB_DELAY_SEC = 2.0  # hard pause before every WB CDX call → 0.5 req/s max
WB_429_MAX_RETRIES = 3

# Global 429 backoff — shared across all Wayback coroutines
_wb_backoff_until: float = 0.0
_wb_backoff_lock = asyncio.Lock()


async def _wb_throttle():
    """Sleep until any active 429 backoff window has elapsed."""
    while True:
        wait = _wb_backoff_until - time.monotonic()
        if wait <= 0:
            return
        await asyncio.sleep(wait)


async def _wb_set_backoff(retry_after: str | None, domain: str):
    """On 429: set a global backoff timestamp so ALL Wayback requests pause."""
    global _wb_backoff_until
    async with _wb_backoff_lock:
        secs = int(retry_after) if retry_after and retry_after.isdigit() else 60
        new_until = time.monotonic() + secs
        if new_until > _wb_backoff_until:
            _wb_backoff_until = new_until
            logger.warning("Wayback 429 for %s — ALL WB requests paused for %ds", domain, secs)
    await _wb_throttle()


# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("fetch_html.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


domains_to_remove = [
    # Platforms / hosting / website builders
    'wordpress.com',
    'wpcomstaging.com',
    'wixsite.com',
    'myshopify.com',
    'weebly.com',
    'mybluehost.me',
    'site123.me',
    'ueni.com',

    # Large international corporations
    'marriott.com',
    'hilton.com',
    'ihg.com',
    'mcdonalds.com',
    'mcdonalds.co.uk',
    'subway.com',
    'subway.co.uk',
    'dominos.co.uk',
    'kfc.com',
    'deliveroo.co.uk',
    'sap.com',
    'wipro.com',
    'genpact.com',
    'indeed.com',
    'specsavers.co.uk',
    'foxtons.co.uk',
    'stagecoachbus.com',
    'h10hotels.com',
    'clarks.co.uk',
    'davidlloyd.co.uk',
    'hc-one.co.uk',
    'mol.co.jp',
    'sportradar.com',
    'coherent.com',
    'idexx.com',
    'hexcel.com',
    'veracode.com',
    'digitalriver.com',
    'cmegroup.com',
    'oneill.com',
    'plantronics.com',
    'axa.co.uk',
    'edwards.com',

    # Government / public sector / NHS
    'companieshouse.gov.uk',
    'www.gov.uk',
    'service.gov.uk',
    'www.nhs.uk',
    'nuh.nhs.uk',
    'greenbrook.nhs.uk',
    'cornwallft.nhs.uk',
    'astonhealth.nhs.uk',
    'harnesspcn.nhs.uk',
    'rockhealthcare.nhs.uk',

    # Generic / placeholder TLDs
    'uk.com',
    'uk.net',
    'eu.com',
    'gb.net',
    'rt.com',
    'twitter.com',
    'outlook.com',
    'yell.com',

    # Social media / marketplaces / directories
    'dice.fm',
    'tombola.co.uk',
    'resdiary.com',
    'google.com',
    'linkedin.com',

    # Non-UK domains (Australia, NZ, Israel, Italy, Norway, Ireland, Canada, South Africa)
    'accesscommunity.org.au',
    'astoriagroup.com.au',
    'cellularasset.com.au',
    'endlessspa.com.au',
    'jaymel.com.au',
    'becausewecare.com.au',
    'megaair.co.nz',
    'phantom.co.il',
    'borgolucignanello.com',
    'baglionihotels.com',
    'sweetmahal.no',
    'mccolgans.ie',
    'unitedcleaningservices.ie',
    'carewest.ca',
    'icondesigns.co.za',

    # Regulators / charities
    'salvationarmy.org.uk',
    'cqc.org.uk',
    'nsi.org.uk',
    'careinspectorate.com',

    # Other noisy domains
    'page.com',
    'guess.eu',
    'crocs.co.uk',
    'simon-kucher.com',
    'slalom.com',
    'tracxn.com',
    'northdata.com',
    'kompass.com',
    'beauhurst.com',
    'ccpgames.com',
    'songkick.com',
    'busuu.com',
]

_DOMAINS_TO_REMOVE_SET = set(domains_to_remove)


# ── Domain normalisation ──────────────────────────────────────────────────────

def normalize_domain(raw: str) -> str | None:
    """Return the registered domain (e.g. 'tpgplc.com') or None."""
    if not isinstance(raw, str):
        return None
    raw = raw.strip().lower()
    if not raw:
        return None
    if "://" not in raw:
        raw = "http://" + raw
    hostname = urlparse(raw).netloc or urlparse(raw).path.split("/")[0]
    hostname = hostname.split(":")[0]
    ext = tldextract.extract(hostname)
    if not ext.domain or not ext.suffix:
        return None
    return f"{ext.domain}.{ext.suffix}"


# ── Common Crawl WARC fetch ──────────────────────────────────────────────────

async def fetch_warc_bytes(
    record: dict,
    semaphore: asyncio.Semaphore,
) -> bytes | None:
    """Range-request the exact WARC slice from S3 — never downloads a full ~1 GB file."""
    filename = record["warc_filename"]
    offset   = int(record["offset"])
    length   = int(record["length"])
    byte_range = f"bytes={offset}-{offset + length - 1}"

    async with semaphore:
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: _s3.get_object(
                    Bucket=CC_S3_BUCKET,
                    Key=filename,
                    Range=byte_range,
                    RequestPayer="requester",
                ),
            )
            return response["Body"].read()
        except Exception as exc:
            logger.error("S3 WARC fetch error for %s: %s", filename, exc)
            return None


# ── HTML extraction ───────────────────────────────────────────────────────────

def extract_html(warc_bytes: bytes) -> str | None:
    """Parse a (gzipped) WARC record and return the HTTP response body."""
    try:
        stream = io.BytesIO(warc_bytes)
        for record in ArchiveIterator(stream):
            if record.rec_type == "response":
                raw = record.content_stream().read()
                for enc in ("utf-8", "latin-1", "cp1252"):
                    try:
                        return raw.decode(enc, errors="replace")
                    except LookupError:
                        continue
    except Exception as exc:
        logger.debug("warcio parse error: %s", exc)

    # Fallback: manual gzip + scan for HTML start tag
    try:
        decompressed = gzip.decompress(warc_bytes)
        for marker in (b"<!DOCTYPE", b"<html", b"<HTML"):
            idx = decompressed.find(marker)
            if idx != -1:
                return decompressed[idx:].decode("utf-8", errors="replace")
    except Exception:
        pass

    return None


# ── Wayback Machine ──────────────────────────────────────────────────────────

async def query_wayback_cdx_multi(
    session: aiohttp.ClientSession,
    domain: str,
    limit: int = 50,
) -> list[dict]:
    """
    Query Wayback CDX for up to `limit` snapshots of a domain.
    Filter bullshit pages, deduplicate by URL path, sort by depth,
    return up to PAGE_CAP records as [{url, timestamp, source}, ...].
    """
    params = {
        "url":       domain,
        "matchType": "domain",
        "output":    "json",
        "limit":     str(limit),
        "from":      WB_DATE_FROM,
        "to":        WB_DATE_TO,
        "filter":    "statuscode:200",
        "fl":        "original,timestamp,mimetype",
        "collapse":  "urlkey",
    }
    for _attempt in range(WB_429_MAX_RETRIES):
        await _wb_throttle()
        await asyncio.sleep(WB_DELAY_SEC)
        try:
            async with session.get(
                WB_CDX_URL, params=params, timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status == 429:
                    await _wb_set_backoff(resp.headers.get("Retry-After"), domain)
                    continue
                if resp.status != 200:
                    logger.warning("Wayback CDX %s: HTTP %d (attempt %d)", domain, resp.status, _attempt + 1)
                    if resp.status >= 500:
                        await asyncio.sleep(WB_DELAY_SEC * 2)
                        continue
                    return []
                data = await resp.json(content_type=None)
        except Exception as exc:
            logger.warning("Wayback CDX error for %s: %s (attempt %d)", domain, exc, _attempt + 1)
            await asyncio.sleep(WB_DELAY_SEC * 2)
            continue

        if not data or len(data) < 2:
            return []

        header = data[0]
        rows   = data[1:]
        orig_i = header.index("original")
        ts_i   = header.index("timestamp")
        mime_i = header.index("mimetype")

        candidates = []
        seen_paths: set[str] = set()
        for row in rows:
            url  = row[orig_i]
            mime = row[mime_i]
            if "html" not in mime:
                continue
            if is_bullshit_page(url):
                continue
            norm_path = urlparse(url).path.rstrip("/") or "/"
            if norm_path in seen_paths:
                continue
            seen_paths.add(norm_path)
            candidates.append({
                "url":       url,
                "timestamp": row[ts_i],
                "source":    "wayback",
            })

        candidates.sort(key=lambda r: (url_depth(r["url"]), r["url"]))
        return candidates[:PAGE_CAP]

    logger.warning("Wayback CDX %s: gave up after %d 429s", domain, WB_429_MAX_RETRIES)
    return []


async def fetch_wayback_html(
    session: aiohttp.ClientSession,
    record: dict,
) -> str | None:
    """
    Fetch the raw archived response from Wayback Machine.
    The `id_` modifier returns the original server response without the
    Wayback toolbar/banner injected into the HTML.
    """
    replay_url = f"{WB_BASE_URL}/{record['timestamp']}id_/{record['url']}"
    for _attempt in range(WB_429_MAX_RETRIES):
        await _wb_throttle()
        try:
            async with session.get(
                replay_url, timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status == 429:
                    await _wb_set_backoff(resp.headers.get("Retry-After"), record["url"])
                    continue
                if resp.status != 200:
                    return None
                return await resp.text(errors="replace")
        except Exception as exc:
            logger.warning("Wayback replay error for %s: %s", record["url"], exc)
            return None
    return None


# ── Per-domain orchestration ─────────────────────────────────────────────────

async def process_domain(
    domain: str,
    pages: list[dict],
    output_dir: Path,
    warc_sem: asyncio.Semaphore,
) -> str:
    """Fetch all CC pages for one domain. Returns status string."""
    domain_dir = output_dir / domain
    domain_dir.mkdir(parents=True, exist_ok=True)

    fetched_pages = []
    for page_record in pages:
        slug = url_to_slug(page_record["url"])
        html_path = domain_dir / f"{slug}.html"

        if html_path.exists():
            fetched_pages.append({**page_record, "slug": slug, "status": "skipped"})
            continue

        warc_bytes = await fetch_warc_bytes(page_record, warc_sem)
        if warc_bytes:
            html = extract_html(warc_bytes)
            if html:
                html_path.write_text(html, encoding="utf-8", errors="replace")
                fetched_pages.append({**page_record, "slug": slug, "status": "ok"})
                logger.debug("%s/%s: saved %d chars", domain, slug, len(html))
                continue

        fetched_pages.append({**page_record, "slug": slug, "status": "failed"})

    # Write per-domain manifest
    (domain_dir / "manifest.json").write_text(
        json.dumps(fetched_pages, indent=2), encoding="utf-8",
    )

    ok_count = sum(1 for p in fetched_pages if p["status"] in ("ok", "skipped"))
    logger.info("%s: %d/%d pages fetched (CC)", domain, ok_count, len(pages))
    return f"ok_cc:{ok_count}/{len(pages)}"


async def process_wayback_domain(
    session: aiohttp.ClientSession,
    domain: str,
    output_dir: Path,
    wb_sem: asyncio.Semaphore,
) -> str:
    """Fetch up to PAGE_CAP pages for a domain from Wayback Machine."""
    domain_dir = output_dir / domain
    domain_dir.mkdir(parents=True, exist_ok=True)

    # Query CDX for multiple pages
    async with wb_sem:
        records = await query_wayback_cdx_multi(session, domain)

    if not records:
        logger.info("%s: not found in Wayback (2017–2018)", domain)
        return "not_found"

    fetched_pages = []
    for record in records:
        slug = url_to_slug(record["url"])
        html_path = domain_dir / f"{slug}.html"

        if html_path.exists():
            fetched_pages.append({**record, "slug": slug, "status": "skipped"})
            continue

        html = await fetch_wayback_html(session, record)
        if html:
            html_path.write_text(html, encoding="utf-8", errors="replace")
            fetched_pages.append({**record, "slug": slug, "status": "ok"})
        else:
            fetched_pages.append({**record, "slug": slug, "status": "failed"})

    # Write per-domain manifest
    (domain_dir / "manifest.json").write_text(
        json.dumps(fetched_pages, indent=2), encoding="utf-8",
    )

    ok_count = sum(1 for p in fetched_pages if p["status"] in ("ok", "skipped"))
    logger.info("%s: %d/%d pages fetched (Wayback)", domain, ok_count, len(records))
    return f"ok_wb:{ok_count}/{len(records)}"


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).resolve().parents[1] / "cc_data" / "manifest.json"),
        help="Path to manifest JSON from process_cc_csv.py (default: %(default)s)",
    )
    parser.add_argument(
        "--input-dir",
        default="/Users/eloireynal/Downloads/json_50plus_employees",
        help="Directory of company JSON files (for Wayback fallback domains)",
    )
    parser.add_argument(
        "--sec",
        action="store_true",
        help="Treat --input-dir as a single SEC company_meta.json file "
             "(schema: {cik: {website: url}}) instead of a directory of CH JSONs",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "cc_data" / "raw_html"),
    )
    parser.add_argument("--concurrency-warc", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load CC manifest ──────────────────────────────────────────────────
    manifest: dict[str, list[dict]] = {}
    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        logger.info("Loaded manifest: %d CC domains", len(manifest))
    else:
        logger.warning("Manifest not found at %s — all domains will use Wayback", args.manifest)

    manifest_domains = set(manifest.keys())

    # ── 2. Load company JSONs → extract all domains ──────────────────────────
    all_company_domains: set[str] = set()

    if args.sec:
        # SEC schema: single JSON file {cik: {website: "https://..."}, ...}
        meta_path = Path(args.input_dir)
        logger.info("Loading SEC company_meta.json from %s", meta_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for cik, info in meta.items():
            url = info.get("website", "")
            if url:
                domain = normalize_domain(url)
                if domain:
                    all_company_domains.add(domain)
    else:
        # Companies House schema: directory of JSONs with identifier.websites[]
        input_dir = Path(args.input_dir)
        json_files = sorted(input_dir.glob("*.json"))
        logger.info("Found %d company JSON files in %s", len(json_files), input_dir)
        for p in json_files:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                websites = data.get("identifier", {}).get("websites", [])
                if websites:
                    domain = normalize_domain(websites[0])
                    if domain:
                        all_company_domains.add(domain)
            except Exception:
                continue

    logger.info("Extracted %d unique company domains", len(all_company_domains))

    # ── 3. Wayback candidates = company_domains − manifest − blocklist ───────
    wayback_candidates = all_company_domains - manifest_domains - _DOMAINS_TO_REMOVE_SET
    logger.info("Wayback fallback candidates: %d domains", len(wayback_candidates))

    # ── 4. Skip already-done domains ─────────────────────────────────────────
    def domain_needs_fetch(domain_dir: Path) -> bool:
        """Return True if the domain has no manifest or no successfully fetched HTML."""
        manifest_file = domain_dir / "manifest.json"
        if not manifest_file.exists():
            return True
        try:
            pages = json.loads(manifest_file.read_text())
            return not any(p.get("status") in ("ok", "skipped") for p in pages)
        except Exception:
            return True

    cc_pending = {
        d: pages for d, pages in manifest.items()
        if domain_needs_fetch(output_dir / d)
    }
    wb_pending = [
        d for d in sorted(wayback_candidates)
        if domain_needs_fetch(output_dir / d)
    ]

    logger.info(
        "Pending: %d CC domains, %d Wayback domains (skipped %d + %d already done)",
        len(cc_pending), len(wb_pending),
        len(manifest) - len(cc_pending),
        len(wayback_candidates) - len(wb_pending),
    )

    if args.dry_run:
        logger.info("[DRY RUN] Would fetch %d CC + %d Wayback domains", len(cc_pending), len(wb_pending))
        return

    if not cc_pending and not wb_pending:
        logger.info("Nothing to do — all domains already fetched")
        return

    # ── 5. Async fetch ───────────────────────────────────────────────────────
    warc_sem = asyncio.Semaphore(args.concurrency_warc)
    wb_sem   = asyncio.Semaphore(1)  # strictly one Wayback CDX call at a time

    connector = aiohttp.TCPConnector(limit=100, limit_per_host=20)
    async with aiohttp.ClientSession(
        connector=connector,
        headers={"User-Agent": "Mozilla/5.0 (compatible; academic-research-bot/1.0)"},
    ) as session:
        # CC domains
        counts: dict[str, int] = {}
        if cc_pending:
            cc_tasks = [
                process_domain(domain, pages, output_dir, warc_sem)
                for domain, pages in cc_pending.items()
            ]
            cc_results = await tqdm_asyncio.gather(*cc_tasks, desc="Fetching CC")
            for r in cc_results:
                key = r.split(":")[0]
                counts[key] = counts.get(key, 0) + 1

        # Wayback domains
        if wb_pending:
            wb_tasks = [
                process_wayback_domain(session, domain, output_dir, wb_sem)
                for domain in wb_pending
            ]
            wb_results = await tqdm_asyncio.gather(*wb_tasks, desc="Fetching Wayback")
            for r in wb_results:
                key = r.split(":")[0]
                counts[key] = counts.get(key, 0) + 1

    logger.info("Summary: %s", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    asyncio.run(main())
