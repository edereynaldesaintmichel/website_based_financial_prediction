#!/usr/bin/env python3
"""
Sanitize 10-K HTML filings: fix broken tables, extract raw text + clean HTML tables.

For each HTML file:
1. Strips scripts, styles, links, HRs, page numbers, boilerplate
2. Runs table sanitizer (fixes colspan/rowspan, merges prefix/suffix columns)
3. Replaces every table with a placeholder, captures its sanitized outerHTML
4. Extracts body.innerText (raw text), splices table HTML back at placeholders

Output: {output}/{stem}.md  — raw text with <table> HTML blocks inline

Usage:
    python sanitize_html.py [--output DIR] [--limit N] [--concurrency N]
"""

import argparse
import asyncio
import re
import sys
from pathlib import Path

from tqdm import tqdm

from config import RAW_HTML_DIR, LLM_SANITIZED_DIR

JS_SANITIZE = (Path(__file__).parent / "sanitize_tables.js").read_text(encoding="utf-8")

# JS injected into every page: strips junk, sanitizes tables, extracts raw text.
PROCESS_JS = (
    """
() => {
    // ── Phase 1: Strip junk ──────────────────────────────────────
    document.querySelectorAll('script, style, a, hr')
        .forEach(el => el.remove());

    document.querySelectorAll('p[align="center"], div[align="center"], font')
        .forEach(el => {
            const t = el.innerText.trim();
            if (/^\\d{1,3}$/.test(t)) el.remove();
        });

    document.querySelectorAll('p, div, span').forEach(el => {
        const t = el.innerText.trim().toLowerCase();
        if (t === 'use these links to rapidly review the document' ||
            t === 'table of contents') {
            el.remove();
        }
    });

    // ── Phase 2: Sanitize tables ─────────────────────────────────
    """
    + JS_SANITIZE
    + """

    // ── Phase 3: Classify tables, strip attrs, extract ─────────────
    const allTopTables = Array.from(document.querySelectorAll('table'))
        .filter(t => !t.parentElement.closest('table'));

    let realIdx = 0;
    const tableHtml = {};

    allTopTables.forEach(t => {
        const trs = [...t.querySelectorAll('tbody tr')];
        const avgTds = trs.length > 0
            ? trs.reduce((s, r) => s + r.querySelectorAll('td').length, 0) / trs.length
            : 0;
        let isReal = trs.length > 5 && avgTds > 2;
        if (isReal) {
            const txt = t.textContent.trim()
                .replace(/(?<![,\\d])\\d{4}(?![,\\d])/g, '').replace(/\\s/g, '');
            isReal = ([...txt.matchAll(/\\d/g)].length) / (txt.length || 1) > 0.1;
        }

        if (isReal) {
            // Strip all attributes from the table and every descendant
            [t, ...t.querySelectorAll('*')].forEach(el => {
                while (el.attributes.length > 0)
                    el.removeAttribute(el.attributes[0].name);
            });
            // Re-add structural attrs (colspan/rowspan) where needed
            t.querySelectorAll('td, th').forEach(cell => {
                if (cell.colSpan > 1) cell.setAttribute('colspan', cell.colSpan);
                if (cell.rowSpan > 1) cell.setAttribute('rowspan', cell.rowSpan);
            });

            // Collapse to single line: strip newlines and runs of whitespace between tags
            tableHtml[realIdx] = t.outerHTML.replace(/\\s*\\n\\s*/g, '').replace(/>\\s+</g, '><');
            const placeholder = document.createTextNode('GLMTABLE' + realIdx + 'GLMTABLE');
            t.parentNode.insertBefore(placeholder, t);
            t.remove();
            realIdx++;
        } else {
            // Degenerate table: replace with its textContent
            const text = document.createTextNode(t.textContent);
            t.parentNode.replaceChild(text, t);
        }
    });

    return { text: document.body.innerText, tables: tableHtml };
}
"""
)

PLACEHOLDER_RE = re.compile(r"GLMTABLE(\d+)GLMTABLE")


def splice_tables(text: str, tables: dict[str, str]) -> str:
    """Replace placeholders with sanitized table HTML."""
    def replace_match(m):
        idx = m.group(1)
        html = tables.get(idx, tables.get(int(idx), ""))
        if html:
            return f"\n\n{html}\n\n"
        return ""

    return PLACEHOLDER_RE.sub(replace_match, text)


async def process_file(
    html_path: Path, browser, sem: asyncio.Semaphore, output_dir: Path, stem: str,
) -> bool:
    """Sanitize one HTML file. Returns True on success."""
    out_path = output_dir / f"{stem}.md"
    if out_path.exists():
        return True

    async with sem:
        page = await browser.new_page(
            viewport={"width": 574, "height": 900}, device_scale_factor=1
        )
        try:
            await page.goto(
                f"file://{html_path.resolve()}",
                wait_until="load",
                timeout=120_000,
            )

            result = await page.evaluate(PROCESS_JS)
            text = result["text"]
            tables = result["tables"]

            # Splice sanitized table HTML back in
            final = splice_tables(text, tables)

            await asyncio.to_thread(out_path.write_text, final, "utf-8")
            return True

        except Exception as e:
            tqdm.write(f"  Error {html_path.name}: {e}")
            return False
        finally:
            await page.close()


async def main():
    parser = argparse.ArgumentParser(
        description="Sanitize 10-K HTML: fix tables, extract raw text + HTML tables"
    )
    parser.add_argument(
        "--input", type=Path, default=RAW_HTML_DIR,
        help=f"Input directory of raw HTML files (default: {RAW_HTML_DIR})",
    )
    parser.add_argument(
        "--output", type=Path, default=LLM_SANITIZED_DIR,
        help=f"Output directory (default: {LLM_SANITIZED_DIR})",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process only first N files")
    parser.add_argument(
        "--concurrency", type=int, default=8,
        help="Max concurrent Playwright pages (default: 8)",
    )
    args = parser.parse_args()

    input_dir = args.input
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        sys.exit(1)

    html_files = sorted(
        f for f in input_dir.rglob("*.html")
        if not f.name.startswith("._")
    )
    if args.limit > 0:
        html_files = html_files[: args.limit]

    if not html_files:
        print(f"No HTML files found in {input_dir}")
        sys.exit(1)

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    already_done = sum(1 for f in html_files if (output_dir / f"{f.stem}.md").exists())

    print(f"Input:   {input_dir} ({len(html_files)} files, {already_done} already done)")
    print(f"Output:  {output_dir}")
    print(f"To do:   {len(html_files) - already_done}")

    if already_done == len(html_files):
        print("All done!")
        return

    from playwright.async_api import async_playwright

    n_ok = 0
    n_err = 0

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        sem = asyncio.Semaphore(args.concurrency)
        pbar = tqdm(total=len(html_files) - already_done, desc="Sanitize", unit="file")

        async def do_one(html_path: Path):
            nonlocal n_ok, n_err
            ok = await process_file(html_path, browser, sem, output_dir, html_path.stem)
            if ok:
                n_ok += 1
            else:
                n_err += 1
            pbar.update(1)

        todo = [f for f in html_files if not (output_dir / f"{f.stem}.md").exists()]
        await asyncio.gather(*[do_one(f) for f in todo])
        pbar.close()
        await browser.close()

    print(f"\nDone! {n_ok} files sanitized, {n_err} errors")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
