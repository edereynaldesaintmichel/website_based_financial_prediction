#!/usr/bin/env python3
"""
Unified HTML -> Markdown pipeline using GLM-OCR via vLLM.

Converts HTML financial filings to clean markdown by rendering them
to PDF (Playwright/Chromium), splitting into page images (PyMuPDF),
and OCR'ing each page with GLM-OCR served by vLLM.

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
    --concurrency N     Max concurrent OCR requests (default: 16)
    --dpi N             DPI for page rendering (default: 200)
    --port N            vLLM server port (default: 8080)
    --no-tag            Skip number tagging
    --no-zip            Skip zipping output
    --no-server         Don't auto-start vLLM (assume it's already running)
"""

import argparse
import asyncio
import base64
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import aiohttp
import fitz  # PyMuPDF
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
# Phase 2: PDF → page images via PyMuPDF
# ──────────────────────────────────────────────────────────────

def pdf_to_images(pdf_path: Path, img_dir: Path, dpi: int = 200) -> list[Path]:
    """Split a PDF into per-page PNG images. Returns list of image paths."""
    img_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    paths = []
    for i in range(len(doc)):
        img_path = img_dir / f"p{i:04d}.png"
        if not img_path.exists():
            pix = doc[i].get_pixmap(dpi=dpi)
            pix.save(str(img_path))
        paths.append(img_path)
    doc.close()
    return paths


# ──────────────────────────────────────────────────────────────
# Phase 3: OCR via vLLM (OpenAI-compatible API)
# ──────────────────────────────────────────────────────────────

async def ocr_page(session: aiohttp.ClientSession, image_path: Path,
                   api_base: str) -> str:
    """OCR a single page image. Uses file:// URL (requires --allowed-local-media-path)."""
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"file://{image_path.resolve()}"},
                },
                {"type": "text", "text": "Text Recognition:"},
            ],
        }],
        "max_tokens": 8192,
    }

    async with session.post(f"{api_base}/chat/completions", json=payload) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"OCR API error {resp.status}: {body[:300]}")
        result = await resp.json()
        return result["choices"][0]["message"]["content"]


async def ocr_document_pages(session: aiohttp.ClientSession,
                             image_paths: list[Path],
                             api_base: str,
                             sem: asyncio.Semaphore,
                             page_pbar: tqdm | None = None) -> list[str]:
    """OCR all pages of a single document with concurrency control."""

    async def do_one(img_path: Path) -> str:
        async with sem:
            result = await ocr_page(session, img_path, api_base)
            if page_pbar:
                page_pbar.update(1)
            return result

    return await asyncio.gather(*[do_one(p) for p in image_paths])


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
    parser.add_argument("--concurrency", type=int, default=16,
                        help="Max concurrent OCR requests")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for page rendering")
    parser.add_argument("--port", type=int, default=8080, help="vLLM server port")
    parser.add_argument("--no-tag", action="store_true", help="Skip number tagging")
    parser.add_argument("--no-zip", action="store_true", help="Skip zipping output")
    parser.add_argument("--no-server", action="store_true",
                        help="Don't auto-start vLLM (assume already running)")
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
    img_dir = tmp_dir / "images"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

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

    # ── Phase 2+3: vLLM server + OCR ────────────────────────
    print(f"\n=== Phase 2+3: PDF → images → OCR ({len(pdf_files)} files) ===")

    vllm_proc = None
    if not args.no_server and not is_server_running(args.port):
        vllm_proc = start_vllm_server(args.port)
    elif is_server_running(args.port):
        print(f"  Using existing vLLM server on port {args.port}")
    else:
        print(f"  ERROR: No vLLM server on port {args.port}. Start one or remove --no-server")
        sys.exit(1)

    api_base = f"http://localhost:{args.port}/v1"
    sem = asyncio.Semaphore(args.concurrency)
    total_pages = 0
    n_done = 0
    n_errors = 0

    try:
        timeout = aiohttp.ClientTimeout(total=600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # First pass: count total pages for the progress bar
            all_page_counts = {}
            for pdf_path in pdf_files:
                stem = pdf_path.stem
                if (output_dir / f"{stem}.md").exists():
                    continue
                doc = fitz.open(str(pdf_path))
                all_page_counts[pdf_path] = len(doc)
                doc.close()

            docs_to_process = [p for p in pdf_files if p in all_page_counts]
            total_page_count = sum(all_page_counts.values())

            file_pbar = tqdm(docs_to_process, desc="Files", unit="file")
            page_pbar = tqdm(total=total_page_count, desc="Pages (OCR)", unit="pg")

            for pdf_path in file_pbar:
                stem = pdf_path.stem
                md_path = output_dir / f"{stem}.md"
                file_pbar.set_postfix_str(stem[:30])

                # PDF → page images
                doc_img_dir = img_dir / stem
                images = pdf_to_images(pdf_path, doc_img_dir, dpi=args.dpi)
                n_pages = len(images)
                total_pages += n_pages

                try:
                    pages_md = await ocr_document_pages(
                        session, images, api_base, sem, page_pbar
                    )

                    # Combine pages with separator
                    full_md = "\n\n---\n\n".join(pages_md)

                    # Tag numbers
                    if not args.no_tag:
                        full_md = tag_numbers_in_text(full_md)

                    md_path.write_text(full_md, encoding="utf-8")
                    n_done += 1
                except Exception as e:
                    n_errors += 1
                    tqdm.write(f"  FAILED {stem}: {e}")

                # Cleanup page images for this document
                shutil.rmtree(doc_img_dir, ignore_errors=True)

            file_pbar.close()
            page_pbar.close()

    finally:
        if vllm_proc:
            print("\nStopping vLLM server...")
            vllm_proc.terminate()
            try:
                vllm_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                vllm_proc.kill()

    elapsed = time.time() - t0

    # ── Phase 4: Zip output ──────────────────────────────────
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
    print(f"\nDone! {n_done} files, {total_pages} pages in {elapsed:.0f}s")
    if total_pages:
        print(f"  {elapsed / total_pages:.2f}s/page, "
              f"{total_pages / elapsed:.1f} pages/s")
    if n_errors:
        print(f"  {n_errors} errors")


if __name__ == "__main__":
    asyncio.run(main())
