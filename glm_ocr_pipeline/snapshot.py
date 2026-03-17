#!/usr/bin/env python3
"""
Phase 1: Screenshot tables from HTML filings + extract plain text.

For each HTML file (in Playwright):
- Strips links, scripts, styles, page numbers, HRs
- Replaces degenerate tables with their innerText
- Replaces real tables with placeholders, screenshots them as PNGs
- Extracts document.body.innerText

Output structure:
    {output}/tables/      — {stem}__t{i}.png
    {output}/text/        — {stem}.md  (with GLMOCRTABLE{i}GLMOCRTABLE placeholders)

Usage:
    python snapshot.py <input_dir_or_zip> [--output DIR] [--limit N]
"""

import argparse
import asyncio
import re
import sys
import zipfile
from pathlib import Path

from tqdm import tqdm

# JS injected into every page: classifies tables, strips junk, extracts text.
PROCESS_JS = """
() => {
    // Remove scripts, styles, links, HRs
    document.querySelectorAll('script, style, a, hr')
        .forEach(el => el.remove());

    // Remove page-number-like elements:
    //   - p[align="center"] with short numeric-ish content
    //   - any block element whose trimmed text is just a number (1-3 digits)
    document.querySelectorAll('p[align="center"], div[align="center"], font')
        .forEach(el => {
            const t = el.innerText.trim();
            if (/^\\d{1,3}$/.test(t)) el.remove();
        });

    // Remove common navigation/boilerplate text
    document.querySelectorAll('p, div, span').forEach(el => {
        const t = el.innerText.trim().toLowerCase();
        if (t === 'use these links to rapidly review the document' ||
            t === 'table of contents') {
            el.remove();
        }
    });

    // Classify top-level tables
    const allTopTables = Array.from(document.querySelectorAll('table'))
        .filter(t => !t.parentElement.closest('table'));

    let realIdx = 0;
    allTopTables.forEach(t => {
        const trs = t.querySelectorAll('tr');
        const tds = t.querySelectorAll('td');
        const isReal = trs.length >= 5 && (tds.length / trs.length) >= 2;
        if (isReal) {
            t.setAttribute('data-table-idx', realIdx);
            const placeholder = document.createTextNode(
                'GLMOCRTABLE' + realIdx + 'GLMOCRTABLE'
            );
            t.parentNode.insertBefore(placeholder, t);
            t.style.display = 'none';
            realIdx++;
        } else {
            // Degenerate table: replace with its innerText
            const text = document.createTextNode(t.innerText);
            t.parentNode.replaceChild(text, t);
        }
    });

    return { nTables: realIdx, text: document.body.innerText };
}
"""


async def process_file(
    html_path: Path, browser, sem: asyncio.Semaphore,
    tables_dir: Path, md_dir: Path, stem: str,
) -> tuple[int, int]:
    """Screenshot real tables and extract plain text. Returns (n_screenshots, n_tables)."""

    text_path = md_dir / f"{stem}.md"
    if text_path.exists():
        return 0, 0

    async with sem:
        page = await browser.new_page(viewport={"width": 1280, "height": 900})
        try:
            await page.goto(
                f"file://{html_path.resolve()}",
                wait_until="load",
                timeout=120_000,
            )

            result = await page.evaluate(PROCESS_JS)
            n_tables = result["nTables"]
            text = result["text"]

            # Strip leading/trailing whitespace per line, drop whitespace-only lines
            lines = text.split("\n")
            lines = [line.strip() for line in lines]
            text = "\n".join(lines)

            # Collapse 3+ consecutive blank lines
            text = re.sub(r"\n{3,}", "\n\n", text)

            # Screenshot real tables
            saved = 0
            for i in range(n_tables):
                out_path = tables_dir / f"{stem}__t{i}.png"
                if out_path.exists():
                    saved += 1
                    continue
                try:
                    el = await page.query_selector(f'table[data-table-idx="{i}"]')
                    if el:
                        await el.evaluate('el => el.style.display = ""')
                        png_bytes = await el.screenshot(type="png", timeout=3000)
                        await asyncio.to_thread(out_path.write_bytes, png_bytes)
                        saved += 1
                except Exception as e:
                    tqdm.write(f"  Screenshot error {stem} table {i}: {e}")

            await asyncio.to_thread(text_path.write_text, text, "utf-8")

        except Exception as e:
            tqdm.write(f"  Error {html_path.name}: {e}")
            return 0, 0
        finally:
            await page.close()

    return saved, n_tables


async def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Screenshot tables + extract plain text"
    )
    parser.add_argument("input", help="Input directory or .zip of HTML files")
    parser.add_argument("--output", help="Output directory (default: {input}_snapshot)")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N files")
    parser.add_argument("--browser-concurrency", type=int, default=8,
                        help="Max concurrent Playwright pages (default: 8)")
    args = parser.parse_args()

    input_path = Path(args.input)

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

    html_files = sorted(
        f for f in input_dir.rglob("*.html")
        if not f.name.startswith("._")
    )
    if args.limit > 0:
        html_files = html_files[:args.limit]

    if not html_files:
        print(f"No HTML files found in {input_dir}")
        sys.exit(1)

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = input_dir.parent / f"{input_dir.name}_snapshot"

    tables_dir = output_dir / "tables"
    md_dir = output_dir / "markdown"
    tables_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {input_dir} ({len(html_files)} files)")
    print(f"Tables: {tables_dir}")
    print(f"Markdown: {md_dir}")

    print(f"\n=== Processing files ===")
    from playwright.async_api import async_playwright

    total_screenshots = 0
    total_tables = 0

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        sem = asyncio.Semaphore(args.browser_concurrency)

        pbar = tqdm(total=len(html_files), desc="Process", unit="file")

        async def process(html_path: Path):
            nonlocal total_screenshots, total_tables
            n_shots, n_t = await process_file(
                html_path, browser, sem, tables_dir, md_dir, html_path.stem,
            )
            total_screenshots += n_shots
            total_tables += n_t
            pbar.update(1)

        await asyncio.gather(*[process(f) for f in html_files])
        pbar.close()
        await browser.close()

    print(f"\nDone! {len(html_files)} files processed")
    print(f"  {total_screenshots} table screenshots → {tables_dir}")
    print(f"  {total_tables} placeholders in text")
    print(f"\nNext: upload {tables_dir}/ to your GPU server and run ocr_tables.py")


if __name__ == "__main__":
    asyncio.run(main())
