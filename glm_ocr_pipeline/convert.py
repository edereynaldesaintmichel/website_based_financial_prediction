#!/usr/bin/env python3
"""
Unified HTML -> Markdown pipeline using GLM-OCR SDK + vLLM.

Converts HTML financial filings to clean markdown by rendering them
to PDF (Playwright/Chromium), then using the GLM-OCR SDK for
layout-aware OCR (PP-DocLayoutV3 → region-specific recognition).

Two-phase-per-batch design to maximise GPU utilisation:
  Phase A: Load PDFs + run layout detection on all pages (GPU: layout model only)
  Phase B: Blast all cropped regions to vLLM at once   (GPU: vLLM only)
No GPU contention — each phase owns the GPU exclusively.

Output is a single .jsonl file where each line is:
    {"stem": "<filename_without_ext>", "markdown": "<content>"}

Lines are written progressively as batches complete, so the output is
safe to inspect mid-run and the pipeline can resume from where it left off.

Usage:
    # 1. Run setup_companies_house_ocr.sh (starts vLLM, installs deps, downloads data)
    #
    # 2. Run the pipeline:
    python3 convert.py <input_dir_or_zip> [options]

    # Quick test on 5 files:
    python3 convert.py raw_html.zip --limit 5

Options:
    --output FILE       Output .jsonl path (default: {input}_cleaned_up.jsonl)
    --limit N           Process only first N files
    --port N            vLLM server port (default: 8001)
    --no-layout         Disable layout detection (faster but no table/formula handling)
"""

import argparse
import asyncio
import json
import os
import shutil
import socket
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import aiohttp

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))


# ──────────────────────────────────────────────────────────────
# Phase 1: HTML → PDF via Playwright
# ──────────────────────────────────────────────────────────────

async def html_to_pdf_batch(html_paths: list[Path], pdf_dir: Path,
                            browser_concurrency: int = 4) -> list[tuple[str, str]]:
    """Convert HTML files to PDFs. Returns [(filename, status), ...]."""
    from playwright.async_api import async_playwright

    pdf_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(browser_concurrency)
    results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        pbar = tqdm(total=len(html_paths), desc="HTML → PDF", unit="file")

        async def convert_one(html_path: Path):
            pdf_path = pdf_dir / f"{html_path.stem}.pdf"
            if pdf_path.exists():
                pbar.update(1)
                return (html_path.name, "skipped")
            async with sem:
                try:
                    page = await browser.new_page()
                    await page.goto(
                        f"file://{html_path.resolve()}",
                        wait_until="load",
                        timeout=120_000,
                    )
                    tmp_path = pdf_path.with_suffix(f".pdf.tmp.{os.getpid()}.{id(html_path)}")
                    await page.pdf(
                        path=str(tmp_path),
                        format="Letter",
                        print_background=True,
                    )
                    os.replace(tmp_path, pdf_path)
                    await page.close()
                    pbar.update(1)
                    return (html_path.name, "ok")
                except Exception as e:
                    pbar.update(1)
                    return (html_path.name, f"error: {e}")

        results = await asyncio.gather(*[convert_one(p) for p in html_paths])
        pbar.close()
        await browser.close()

    return results


def is_server_running(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=3):
            return True
    except OSError:
        return False


# ──────────────────────────────────────────────────────────────
# Phase 2: Two-phase OCR (layout then vLLM, sequential per batch)
# ──────────────────────────────────────────────────────────────

async def run_ocr_two_phase(
    pdf_files: list[Path],
    jsonl_path: Path,
    config_path: str,
    port: int,
    enable_layout: bool,
    batch_size: int = 200,
):
    """Process PDFs in batches. Each batch: layout-detect all pages, then OCR all regions."""
    from glmocr.config import load_config
    from glmocr.dataloader import PageLoader
    from glmocr.ocr_client import OCRClient
    from glmocr.postprocess import ResultFormatter
    from glmocr.utils.image_utils import crop_image_region

    cfg = load_config(config_path)
    pcfg = cfg.pipeline

    page_loader = PageLoader(pcfg.page_loader)
    ocr_client = OCRClient(pcfg.ocr_api)
    ocr_client._pool_maxsize = pcfg.max_workers
    ocr_client.start()

    result_formatter = ResultFormatter(pcfg.result_formatter)

    layout_detector = None
    if enable_layout:
        from glmocr.layout import PPDocLayoutDetector
        layout_detector = PPDocLayoutDetector(pcfg.layout)
        layout_detector.start()

    use_polygon = pcfg.layout.use_polygon if enable_layout else False

    n_done = 0
    n_errors = 0

    pbar = tqdm(total=len(pdf_files), desc="OCR", unit="file")

    try:
        with jsonl_path.open("a", encoding="utf-8") as jsonl_fh:
            for batch_start in range(0, len(pdf_files), batch_size):
                batch = pdf_files[batch_start:batch_start + batch_size]

                # ── Phase A: Load pages + layout detection ──────────────
                unit_pages: dict[int, list] = {}
                batch_sources = [f"file://{p.resolve()}" for p in batch]

                all_pages = []
                for page_img, unit_idx in page_loader.iter_pages_with_unit_indices(batch_sources):
                    all_pages.append((unit_idx, page_img))

                if enable_layout and layout_detector is not None:
                    layout_batch_size = layout_detector.batch_size
                    all_layouts = []

                    for lb_start in range(0, len(all_pages), layout_batch_size):
                        lb_end = lb_start + layout_batch_size
                        lb_images = [img for _, img in all_pages[lb_start:lb_end]]
                        layouts, _ = layout_detector.process(
                            lb_images,
                            save_visualization=False,
                            global_start_idx=lb_start,
                            use_polygon=use_polygon,
                        )
                        all_layouts.extend(layouts)

                    for i, (unit_idx, page_img) in enumerate(all_pages):
                        unit_pages.setdefault(unit_idx, []).append(
                            (page_img, all_layouts[i])
                        )
                else:
                    for unit_idx, page_img in all_pages:
                        w, h = page_img.size
                        fake_region = [{
                            "bbox_2d": [0, 0, w, h],
                            "task_type": "text",
                            "label": "text",
                            "index": 0,
                        }]
                        unit_pages.setdefault(unit_idx, []).append(
                            (page_img, fake_region)
                        )

                # ── Crop all regions ───────────────────────────────────
                ocr_tasks = []
                skip_results = []

                for unit_idx, pages in unit_pages.items():
                    for page_local_idx, (page_img, layout_regions) in enumerate(pages):
                        for region in layout_regions:
                            task_type = region.get("task_type", "text")
                            if task_type == "abandon":
                                continue

                            bbox = region.get("bbox_2d")
                            polygon = region.get("polygon") if use_polygon else None
                            try:
                                cropped = crop_image_region(page_img, bbox, polygon)
                            except Exception as e:
                                tqdm.write(f"  Crop failed unit={unit_idx} bbox={bbox}: {e}")
                                region["content"] = ""
                                skip_results.append((unit_idx, page_local_idx, region, None))
                                continue

                            if task_type == "skip":
                                region["content"] = None
                                skip_results.append((unit_idx, page_local_idx, region, cropped))
                                continue

                            ocr_tasks.append((unit_idx, page_local_idx, region, cropped))

                del all_pages
                for unit_idx in unit_pages:
                    unit_pages[unit_idx] = [(None, layout) for _, layout in unit_pages[unit_idx]]

                # ── Phase B: Save images to disk, then blast vLLM ──────
                tqdm.write(f"  Batch {batch_start//batch_size + 1}: "
                           f"{len(ocr_tasks)} regions to OCR across "
                           f"{len(unit_pages)} files")

                results_by_unit: dict[int, dict[int, list]] = {}
                for unit_idx, page_local_idx, region, cropped in skip_results:
                    results_by_unit.setdefault(unit_idx, {}).setdefault(
                        page_local_idx, []
                    ).append(region)

                region_dir = Path(tempfile.mkdtemp(prefix="ocr_regions_"))
                prebuilt = []
                encode_pbar = tqdm(
                    total=len(ocr_tasks), desc="  Save", unit="region", leave=False
                )
                img_format = page_loader.image_format
                for i, (unit_idx, page_local_idx, region, cropped) in enumerate(ocr_tasks):
                    task_type = region.get("task_type", "text")
                    prompt_text = ""
                    if page_loader.task_prompt_mapping:
                        prompt_text = page_loader.task_prompt_mapping.get(task_type, "")

                    img_path = region_dir / f"r{i}.jpg"
                    cropped.save(str(img_path), format=img_format)

                    content = [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"file://{img_path}"},
                        },
                    ]
                    if prompt_text:
                        content.append({"type": "text", "text": prompt_text})

                    req = {
                        "model": ocr_client.model or "glm-ocr",
                        "messages": [{"role": "user", "content": content}],
                        "max_tokens": page_loader.max_tokens,
                        "temperature": page_loader.temperature,
                        "top_p": page_loader.top_p,
                        "top_k": page_loader.top_k,
                        "repetition_penalty": page_loader.repetition_penalty,
                    }
                    serialized = json.dumps(req).encode("utf-8")
                    prebuilt.append((unit_idx, page_local_idx, region, serialized))
                    encode_pbar.update(1)
                encode_pbar.close()

                del ocr_tasks

                # Blast all requests via aiohttp
                api_url = ocr_client.api_url
                api_headers = {"Content-Type": "application/json"}
                if ocr_client.api_key:
                    api_headers["Authorization"] = f"Bearer {ocr_client.api_key}"
                api_headers.update(ocr_client.extra_headers)
                api_timeout_s = ocr_client.request_timeout

                async def _blast_vllm(tasks):
                    connector = aiohttp.TCPConnector(limit=0, ssl=False)
                    timeout = aiohttp.ClientTimeout(total=api_timeout_s)
                    ocr_pbar = tqdm(
                        total=len(tasks), desc="  vLLM", unit="region", leave=False
                    )

                    async with aiohttp.ClientSession(
                        connector=connector, timeout=timeout
                    ) as session:
                        sem = asyncio.Semaphore(512)

                        async def _do_one(unit_idx, page_local_idx, region, data_bytes):
                            async with sem:
                                try:
                                    async with session.post(
                                        api_url,
                                        headers=api_headers,
                                        data=data_bytes,
                                    ) as resp:
                                        if resp.status == 200:
                                            result = await resp.json()
                                            content = result["choices"][0]["message"]["content"]
                                            region["content"] = content.strip() if content else ""
                                        else:
                                            region["content"] = None
                                except Exception:
                                    region["content"] = None
                                ocr_pbar.update(1)
                                return unit_idx, page_local_idx, region

                        results = await asyncio.gather(
                            *[_do_one(u, p, r, d) for u, p, r, d in tasks]
                        )

                    ocr_pbar.close()
                    return results

                try:
                    ocr_results = await _blast_vllm(prebuilt)
                finally:
                    shutil.rmtree(region_dir, ignore_errors=True)

                for unit_idx, page_local_idx, region in ocr_results:
                    results_by_unit.setdefault(unit_idx, {}).setdefault(
                        page_local_idx, []
                    ).append(region)

                # ── Phase C: Format results and write JSONL ─────────────
                for unit_idx, pdf_path in enumerate(batch):
                    stem = pdf_path.stem
                    page_results = results_by_unit.get(unit_idx, {})

                    if not page_results:
                        n_errors += 1
                        tqdm.write(f"  No results for {stem}")
                        pbar.update(1)
                        continue

                    n_pages = len(unit_pages.get(unit_idx, []))
                    grouped = []
                    for page_local_idx in range(n_pages):
                        regions = page_results.get(page_local_idx, [])
                        grouped.append(regions)

                    try:
                        _, md_result, _ = result_formatter.process(
                            grouped, cropped_images=None
                        )
                        jsonl_fh.write(
                            json.dumps(
                                {"stem": stem, "markdown": md_result},
                                ensure_ascii=False,
                            ) + "\n"
                        )
                        jsonl_fh.flush()
                        n_done += 1
                    except Exception as e:
                        n_errors += 1
                        tqdm.write(f"  FAILED {stem}: {e}")

                    pbar.update(1)

    finally:
        pbar.close()
        ocr_client.stop()
        if layout_detector is not None:
            layout_detector.stop()

    return n_done, n_errors


# ──────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description="Convert HTML filings to markdown via GLM-OCR"
    )
    parser.add_argument("input", help="Input directory or .zip of HTML files")
    parser.add_argument("--output", help="Output .jsonl path (default: {input}_cleaned_up.jsonl)")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N files")
    parser.add_argument("--port", type=int, default=8001, help="vLLM server port")
    parser.add_argument("--no-layout", action="store_true",
                        help="Disable layout detection (faster but worse tables)")
    args = parser.parse_args()

    input_path = Path(args.input)

    # ── Unzip if needed ──────────────────────────────────────
    if input_path.suffix == ".zip":
        extract_dir = input_path.with_suffix("")
        if not extract_dir.exists():
            print(f"Extracting {input_path} ...")
            with zipfile.ZipFile(input_path) as zf:
                zf.extractall(extract_dir)
        input_dir = extract_dir
    else:
        input_dir = input_path

    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        sys.exit(1)

    # ── Discover HTML files ──────────────────────────────────
    html_files = sorted(
        f for f in input_dir.rglob("*.html")
        if not f.name.startswith("._")
    )
    if args.limit > 0:
        html_files = html_files[:args.limit]

    if not html_files:
        print(f"No HTML files found in {input_dir}")
        sys.exit(1)

    # ── Output JSONL path ─────────────────────────────────────
    if args.output:
        jsonl_path = Path(args.output)
    else:
        name = input_dir.name.rstrip("/")
        jsonl_path = input_dir.parent / f"{name}_cleaned_up.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip already-processed files
    existing: set[str] = set()
    if jsonl_path.exists():
        with jsonl_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        existing.add(json.loads(line)["stem"])
                    except (json.JSONDecodeError, KeyError):
                        pass

    todo = [f for f in html_files if f.stem not in existing]

    print(f"Input:  {input_dir}  ({len(html_files)} HTML files, {len(existing)} already done)")
    print(f"Output: {jsonl_path}")

    if not todo:
        print("All files already processed!")
        return

    print(f"To process: {len(todo)} files\n")

    # ── PDF cache ────────────────────────────────────────────
    pdf_dir = Path(os.environ.get("GLM_OCR_PDF_CACHE", "/workspace/pdf_cache"))
    pdf_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ── Phase 1: HTML → PDF ──────────────────────────────────
    print("=== Phase 1: HTML → PDF (Playwright) ===")
    results = await html_to_pdf_batch(todo, pdf_dir)
    n_ok = sum(1 for _, s in results if s == "ok")
    n_skip = sum(1 for _, s in results if s == "skipped")
    errors = [(n, s) for n, s in results if s.startswith("error")]
    print(f"  {n_ok} converted, {n_skip} skipped, {len(errors)} errors")
    for name, err in errors[:10]:
        print(f"    {name}: {err}")

    successful_stems = {
        Path(n).stem for n, s in results if s in ("ok", "skipped")
    }
    pdf_files = sorted(
        p for p in pdf_dir.glob("*.pdf") if p.stem in successful_stems
    )

    if not pdf_files:
        print("No PDFs to process!")
        return

    # ── Phase 2: Two-phase OCR ───────────────────────────────
    enable_layout = not args.no_layout
    mode = "layout-aware" if enable_layout else "OCR-only"
    print(f"\n=== Phase 2: Two-phase OCR ({mode}, {len(pdf_files)} files) ===")

    if not is_server_running(args.port):
        print(f"  ERROR: No vLLM server on port {args.port}. Run setup_companies_house_ocr.sh first.")
        sys.exit(1)
    print(f"  Using vLLM server on port {args.port}")

    config_path = str(Path(__file__).parent / "config.yaml")

    n_done, n_errors = await run_ocr_two_phase(
        pdf_files=pdf_files,
        jsonl_path=jsonl_path,
        config_path=config_path,
        port=args.port,
        enable_layout=enable_layout,
    )

    elapsed = time.time() - t0
    size_mb = jsonl_path.stat().st_size / 1024 / 1024
    print(f"\nDone! {n_done} files in {elapsed:.0f}s → {jsonl_path} ({size_mb:.1f} MB)")
    if n_errors:
        print(f"  {n_errors} errors")


if __name__ == "__main__":
    asyncio.run(main())
