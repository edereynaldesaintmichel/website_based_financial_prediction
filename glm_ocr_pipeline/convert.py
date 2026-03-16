#!/usr/bin/env python3
"""
HTML → Markdown pipeline for SEC filings (hybrid approach).

- Text/headings/lists: converted directly via markdownify (instant)
- Tables: screenshotted via Playwright, OCR'd by GLM-OCR on vLLM

Only table images hit the GPU — no PDF conversion, no layout detection.

Usage:
    # 1. Start vLLM server (or let this script auto-start it):
    #    vllm serve zai-org/GLM-OCR --served-model-name glm-ocr --port 8000 \
    #        --gpu-memory-utilization 0.95 --max-num-seqs 128 \
    #        --enable-prefix-caching --enable-chunked-prefill --dtype bfloat16
    #
    # 2. Run the pipeline:
    python convert.py <input_dir_or_zip> [options]

Options:
    --output DIR              Output directory (default: {input}_cleaned_up)
    --limit N                 Process only first N files
    --port N                  vLLM server port (default: 8000)
    --no-zip                  Skip zipping output
    --no-server               Don't auto-start vLLM (assume it's already running)
    --keep-server             Don't kill vLLM server when done
    --concurrency N           Max concurrent vLLM requests (default: 64)
    --browser-concurrency N   Max concurrent Playwright pages (default: 8)
"""

import argparse
import asyncio
import base64
import re
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import aiohttp
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

MODEL_ID = "zai-org/GLM-OCR"
SERVED_MODEL_NAME = "glm-ocr"
PLACEHOLDER = "___TABLE_{}_PLACEHOLDER___"


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
            "--model", MODEL_ID,
            "--served-model-name", SERVED_MODEL_NAME,
            "--port", str(port),
            "--gpu-memory-utilization", "0.95",
            "--max-num-seqs", "128",
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--dtype", "bfloat16",
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
# Phase 1: Screenshot tables via Playwright
# ──────────────────────────────────────────────────────────────

async def screenshot_tables_for_file(
    html_path: Path, browser, sem: asyncio.Semaphore,
) -> list[bytes | None]:
    """Screenshot top-level <table> elements from one HTML file."""
    html_text = html_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html_text, "html.parser")
    n_tables = len([t for t in soup.find_all("table") if not t.find_parent("table")])

    if n_tables == 0:
        return []

    screenshots: list[bytes | None] = []
    async with sem:
        page = await browser.new_page(viewport={"width": 1280, "height": 900})
        try:
            await page.goto(
                f"file://{html_path.resolve()}",
                wait_until="load",
                timeout=120_000,
            )

            # Tag top-level tables with indices in the browser DOM
            n_browser = await page.evaluate("""
                () => {
                    const tables = Array.from(document.querySelectorAll('table'))
                        .filter(t => !t.parentElement.closest('table'));
                    tables.forEach((t, i) => t.setAttribute('data-table-idx', i));
                    return tables.length;
                }
            """)

            if n_browser != n_tables:
                tqdm.write(
                    f"  Warning: {html_path.name}: BS4 found {n_tables} tables, "
                    f"Playwright found {n_browser}"
                )

            for i in range(min(n_tables, n_browser)):
                try:
                    el = await page.query_selector(f'table[data-table-idx="{i}"]')
                    if el:
                        screenshots.append(await el.screenshot(type="png"))
                    else:
                        screenshots.append(None)
                except Exception:
                    screenshots.append(None)
        except Exception as e:
            tqdm.write(f"  Playwright error for {html_path.name}: {e}")
            screenshots = [None] * n_tables
        finally:
            await page.close()

    return screenshots


# ──────────────────────────────────────────────────────────────
# Phase 2: OCR tables via vLLM
# ──────────────────────────────────────────────────────────────

async def ocr_all_tables(
    table_images: list[tuple[int, int, bytes]],
    port: int,
    concurrency: int,
) -> dict[tuple[int, int], str]:
    """Send table images to vLLM for OCR. Returns {(file_idx, table_idx): markdown}."""
    results: dict[tuple[int, int], str] = {}
    sem = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession() as session:
        pbar = tqdm(total=len(table_images), desc="OCR tables", unit="table")

        async def ocr_one(file_idx: int, table_idx: int, img: bytes):
            b64 = base64.b64encode(img).decode()
            payload = {
                "model": SERVED_MODEL_NAME,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                        {"type": "text", "text": "Table Recognition:"},
                    ],
                }],
                "max_tokens": 8192,
                "temperature": 0.1,
            }
            async with sem:
                try:
                    async with session.post(
                        f"http://localhost:{port}/v1/chat/completions",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=180),
                    ) as resp:
                        if resp.status != 200:
                            tqdm.write(
                                f"  vLLM {resp.status} for file {file_idx} table {table_idx}"
                            )
                            return
                        data = await resp.json()
                        results[(file_idx, table_idx)] = (
                            data["choices"][0]["message"]["content"]
                        )
                except Exception as e:
                    tqdm.write(f"  OCR error file {file_idx} table {table_idx}: {e}")
                finally:
                    pbar.update(1)

        await asyncio.gather(*[ocr_one(fi, ti, img) for fi, ti, img in table_images])
        pbar.close()

    return results


# ──────────────────────────────────────────────────────────────
# Phase 3: Assemble markdown
# ──────────────────────────────────────────────────────────────

def assemble_markdown(
    html_path: Path, file_idx: int, ocr_results: dict[tuple[int, int], str],
) -> str:
    """Replace tables with OCR results, convert rest to markdown."""
    html_text = html_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html_text, "html.parser")

    # Strip scripts/styles — not useful for financial text
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()

    top_tables = [t for t in soup.find_all("table") if not t.find_parent("table")]
    table_originals = [str(t) for t in top_tables]

    # Replace tables with unique placeholders
    for i, table in enumerate(top_tables):
        table.replace_with(soup.new_string(PLACEHOLDER.format(f"{file_idx}_{i}")))

    # Convert the (now table-free) HTML to markdown
    markdown = md(str(soup), heading_style="ATX")

    # Splice OCR'd tables (or markdownify fallback) back in
    for i in range(len(top_tables)):
        placeholder = PLACEHOLDER.format(f"{file_idx}_{i}")
        key = (file_idx, i)
        if key in ocr_results:
            replacement = ocr_results[key]
        else:
            # Fallback: let markdownify attempt the table
            replacement = md(table_originals[i])
        markdown = markdown.replace(placeholder, f"\n\n{replacement}\n\n")

    # Collapse excessive blank lines
    markdown = re.sub(r"\n{4,}", "\n\n\n", markdown)

    return markdown


# ──────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description="Convert HTML filings to markdown (markdownify + GLM-OCR for tables)"
    )
    parser.add_argument("input", help="Input directory or .zip of HTML files")
    parser.add_argument("--output", help="Output directory (default: {input}_cleaned_up)")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N files")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--no-zip", action="store_true", help="Skip zipping output")
    parser.add_argument("--no-server", action="store_true",
                        help="Don't auto-start vLLM (assume already running)")
    parser.add_argument("--keep-server", action="store_true",
                        help="Don't kill vLLM server when done")
    parser.add_argument("--concurrency", type=int, default=64,
                        help="Max concurrent vLLM requests (default: 64)")
    parser.add_argument("--browser-concurrency", type=int, default=8,
                        help="Max concurrent Playwright pages (default: 8)")
    args = parser.parse_args()

    input_path = Path(args.input)

    # ── Unzip if needed ──────────────────────────────────────
    if input_path.suffix == ".zip":
        extract_dir = input_path.with_suffix("")
        if not extract_dir.exists():
            print(f"Extracting {input_path}...")
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

    t0 = time.time()

    # ── Phase 1: Screenshot tables ───────────────────────────
    print("=== Phase 1: Screenshot tables (Playwright) ===")
    from playwright.async_api import async_playwright

    all_screenshots: dict[int, list[bytes | None]] = {}
    table_images: list[tuple[int, int, bytes]] = []  # (file_idx, table_idx, png)

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        sem = asyncio.Semaphore(args.browser_concurrency)

        pbar = tqdm(total=len(todo), desc="Screenshots", unit="file")

        async def process_one(file_idx: int, html_path: Path):
            screenshots = await screenshot_tables_for_file(html_path, browser, sem)
            all_screenshots[file_idx] = screenshots
            for i, img in enumerate(screenshots):
                if img is not None:
                    table_images.append((file_idx, i, img))
            pbar.update(1)

        await asyncio.gather(*[process_one(i, f) for i, f in enumerate(todo)])
        pbar.close()
        await browser.close()

    n_tables = len(table_images)
    print(f"  Found {n_tables} tables to OCR across {len(todo)} files")

    # ── Phase 2: OCR tables via vLLM ─────────────────────────
    vllm_proc = None
    ocr_results: dict[tuple[int, int], str] = {}

    if n_tables > 0:
        print(f"\n=== Phase 2: OCR tables via vLLM ({n_tables} tables) ===")

        if not args.no_server and not is_server_running(args.port):
            vllm_proc = start_vllm_server(args.port)
        elif is_server_running(args.port):
            print(f"  Using existing vLLM server on port {args.port}")
        else:
            print(f"  ERROR: No vLLM server on port {args.port}. "
                  "Start one or remove --no-server")
            sys.exit(1)

        try:
            ocr_results = await ocr_all_tables(table_images, args.port, args.concurrency)
        finally:
            if vllm_proc and not args.keep_server:
                print("\nStopping vLLM server...")
                vllm_proc.terminate()
                try:
                    vllm_proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    vllm_proc.kill()
            elif vllm_proc:
                print("\nvLLM server left running (--keep-server)")
    else:
        print("\nNo tables found — skipping vLLM")

    # ── Phase 3: Assemble markdown ───────────────────────────
    print(f"\n=== Phase 3: Assemble markdown ===")
    n_done = 0
    n_errors = 0

    for file_idx, html_path in enumerate(tqdm(todo, desc="Assemble", unit="file")):
        try:
            markdown = assemble_markdown(html_path, file_idx, ocr_results)
            (output_dir / f"{html_path.stem}.md").write_text(markdown, encoding="utf-8")
            n_done += 1
        except Exception as e:
            n_errors += 1
            tqdm.write(f"  FAILED {html_path.stem}: {e}")

    elapsed = time.time() - t0

    # ── Zip output ───────────────────────────────────────────
    if not args.no_zip:
        zip_path = output_dir.with_suffix(".zip")
        print(f"\nZipping → {zip_path}")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for md_file in sorted(output_dir.glob("*.md")):
                zf.write(md_file, md_file.name)
        size_mb = zip_path.stat().st_size / 1024 / 1024
        print(f"  {zip_path} ({size_mb:.1f} MB)")

    # ── Summary ──────────────────────────────────────────────
    n_ocr = len(ocr_results)
    print(f"\nDone! {n_done} files in {elapsed:.0f}s "
          f"({n_tables} tables found, {n_ocr} OCR'd successfully)")
    if n_errors:
        print(f"  {n_errors} file errors")


if __name__ == "__main__":
    asyncio.run(main())
