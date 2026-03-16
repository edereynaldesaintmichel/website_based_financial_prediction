#!/usr/bin/env python3
"""
Unified HTML -> Markdown pipeline using GLM-OCR SDK + vLLM.

Converts HTML financial filings to clean markdown by rendering them
to PDF (Playwright/Chromium), then using the GLM-OCR SDK for
layout-aware OCR (PP-DocLayoutV3 → region-specific recognition).

The SDK handles: PDF → page images → layout detection → parallel
region OCR with task-specific prompts (Text/Table/Formula Recognition)
→ structured Markdown + JSON output.

Usage:
    # 1. Start vLLM server (or let this script auto-start it):
    #    vllm serve zai-org/GLM-OCR --allowed-local-media-path / --port 8080
    #
    # 2. Run the pipeline:
    python convert.py <input_dir_or_zip> [options]

    # Quick test on 5 files:
    python convert.py raw_html.zip --limit 5

Options:
    --output DIR        Output directory (default: {input}_cleaned_up)
    --limit N           Process only first N files
    --port N            vLLM server port (default: 8080)
    --no-tag            Skip number tagging
    --no-zip            Skip zipping output
    --no-server         Don't auto-start vLLM (assume it's already running)
    --no-layout         Disable layout detection (faster but no table/formula handling)
"""

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

# Allow running as `python convert.py` from any directory
sys.path.insert(0, str(Path(__file__).parent))
from tag_numbers import tag_numbers_in_text

MODEL = "zai-org/GLM-OCR"


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
                    await page.pdf(
                        path=str(pdf_path),
                        format="Letter",
                        print_background=True,
                    )
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


# ──────────────────────────────────────────────────────────────
# vLLM server management
# ──────────────────────────────────────────────────────────────

def is_server_running(port: int) -> bool:
    try:
        urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3)
        return True
    except Exception:
        return False


def start_vllm_server(port: int) -> subprocess.Popen:
    """Start vLLM in the background. Blocks until the server is ready."""
    print(f"Starting vLLM server on port {port} (this may take a few minutes)...")
    log_out = open("vllm_stdout.log", "w")
    log_err = open("vllm_stderr.log", "w")

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL,
            "--allowed-local-media-path", "/",
            "--port", str(port),
        ],
        stdout=log_out,
        stderr=log_err,
    )

    for i in range(600):  # 10 min timeout (model download on first run)
        if proc.poll() is not None:
            log_err.close()
            err_tail = Path("vllm_stderr.log").read_text()[-500:]
            raise RuntimeError(
                f"vLLM exited with code {proc.returncode}.\n"
                f"Last stderr:\n{err_tail}"
            )
        if is_server_running(port):
            print(f"  vLLM ready (took {i}s)")
            return proc
        time.sleep(1)

    proc.kill()
    raise RuntimeError("vLLM failed to start within 10 minutes. Check vllm_stderr.log")


# ──────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description="Convert HTML filings to markdown via GLM-OCR"
    )
    parser.add_argument("input", help="Input directory or .zip of HTML files")
    parser.add_argument("--output", help="Output directory (default: {input}_cleaned_up)")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N files")
    parser.add_argument("--port", type=int, default=8080, help="vLLM server port")
    parser.add_argument("--no-tag", action="store_true", help="Skip number tagging")
    parser.add_argument("--no-zip", action="store_true", help="Skip zipping output")
    parser.add_argument("--no-server", action="store_true",
                        help="Don't auto-start vLLM (assume already running)")
    parser.add_argument("--keep-server", action="store_true",
                        help="Don't kill vLLM server when done")
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
        if not f.name.startswith("._")  # skip macOS resource fork files
    )
    if args.limit > 0:
        html_files = html_files[:args.limit]

    if not html_files:
        print(f"No HTML files found in {input_dir}")
        sys.exit(1)

    # ── Output directory ─────────────────────────────────────
    if args.output:
        output_dir = Path(args.output)
    else:
        name = input_dir.name.rstrip("/")
        output_dir = input_dir.parent / f"{name}_cleaned_up"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip already-processed files
    existing = {p.stem for p in output_dir.glob("*.md")}
    todo = [f for f in html_files if f.stem not in existing]

    print(f"Input:  {input_dir}  ({len(html_files)} HTML files, {len(existing)} already done)")
    print(f"Output: {output_dir}")

    if not todo:
        print("All files already processed!")
        return

    print(f"To process: {len(todo)} files\n")

    # ── Temp directories ─────────────────────────────────────
    tmp_dir = Path("_glm_ocr_tmp")
    pdf_dir = tmp_dir / "pdfs"
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

    # Collect successfully converted PDFs
    successful_stems = {
        Path(n).stem for n, s in results if s in ("ok", "skipped")
    }
    pdf_files = sorted(
        p for p in pdf_dir.glob("*.pdf") if p.stem in successful_stems
    )

    if not pdf_files:
        print("No PDFs to process!")
        return

    # ── Phase 2: vLLM server + GLM-OCR SDK ───────────────────
    enable_layout = not args.no_layout
    mode = "layout-aware" if enable_layout else "OCR-only"
    print(f"\n=== Phase 2: GLM-OCR SDK ({mode}, {len(pdf_files)} files) ===")

    vllm_proc = None
    if not args.no_server and not is_server_running(args.port):
        vllm_proc = start_vllm_server(args.port)
    elif is_server_running(args.port):
        print(f"  Using existing vLLM server on port {args.port}")
    else:
        print(f"  ERROR: No vLLM server on port {args.port}. Start one or remove --no-server")
        sys.exit(1)

    # Initialize GLM-OCR SDK in self-hosted mode
    from glmocr import GlmOcr

    config_path = str(Path(__file__).parent / "config.yaml")
    parser_ocr = GlmOcr(
        config_path=config_path,
        mode="selfhosted",
        ocr_api_host="localhost",
        ocr_api_port=args.port,
        enable_layout=enable_layout,
    )

    n_done = 0
    n_errors = 0

    # Process PDFs in batches: large enough to keep the GPU saturated
    # (many pages/regions in the pipeline at once), small enough to
    # bound RAM (the SDK holds all page images in memory per parse call).
    BATCH_SIZE = 20
    path_to_stem = {str(p): p.stem for p in pdf_files}

    try:
        pbar = tqdm(total=len(pdf_files), desc="OCR", unit="file")
        for batch_start in range(0, len(pdf_files), BATCH_SIZE):
            batch = pdf_files[batch_start:batch_start + BATCH_SIZE]
            batch_strs = [str(p) for p in batch]

            for result in parser_ocr.parse(batch_strs, stream=True):
                source = result.original_images[0] if result.original_images else None
                stem = path_to_stem.get(source)

                if stem is None:
                    n_errors += 1
                    tqdm.write(f"  Could not map result back to source file")
                    pbar.update(1)
                    continue

                try:
                    full_md = result.markdown_result
                    if hasattr(result, '_error') and result._error:
                        raise RuntimeError(result._error)

                    if not args.no_tag:
                        full_md = tag_numbers_in_text(full_md)

                    (output_dir / f"{stem}.md").write_text(full_md, encoding="utf-8")
                    n_done += 1
                except Exception as e:
                    n_errors += 1
                    tqdm.write(f"  FAILED {stem}: {e}")

                pbar.update(1)
        pbar.close()

    finally:
        parser_ocr.close()

        if vllm_proc and not args.keep_server:
            print("\nStopping vLLM server...")
            vllm_proc.terminate()
            try:
                vllm_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                vllm_proc.kill()
        elif vllm_proc:
            print("\nvLLM server left running (--keep-server)")

    elapsed = time.time() - t0

    # ── Phase 3: Zip output ──────────────────────────────────
    if not args.no_zip:
        zip_path = output_dir.with_suffix(".zip")
        print(f"\nZipping → {zip_path}")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for md_file in sorted(output_dir.glob("*.md")):
                zf.write(md_file, md_file.name)
        size_mb = zip_path.stat().st_size / 1024 / 1024
        print(f"  {zip_path} ({size_mb:.1f} MB)")

    # ── Cleanup tmp ──────────────────────────────────────────
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Summary ──────────────────────────────────────────────
    print(f"\nDone! {n_done} files in {elapsed:.0f}s")
    if n_errors:
        print(f"  {n_errors} errors")


if __name__ == "__main__":
    asyncio.run(main())
