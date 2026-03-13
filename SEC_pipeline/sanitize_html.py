"""
Sanitize SEC 10-K HTML filings for LLM consumption.

Adapted from the UK Companies House sanitizer. SEC 10-K HTML filings
differ from iXBRL in that they typically:
  - Have more complex CSS and JavaScript
  - Use <font> tags extensively (older filings)
  - Contain embedded images (base64 or linked)
  - Include EDGAR-specific header/footer boilerplate
  - May have XBRL inline tags (ix:*) in newer filings

Techniques applied (similar to UK pipeline):
1. Remove EDGAR header/footer boilerplate.
2. Remove <script>, <style> blocks.
3. Remove hidden elements (display:none).
4. Unwrap XBRL inline tags and formatting wrappers.
5. Strip attributes (keep only colspan, rowspan, href).
6. Purge empty elements.
7. Flatten degenerate (single-column, no-header) tables.
8. Flatten single-child nesting.
9. Normalize entities and whitespace.
10. Extract body content only.

Usage:
    python sanitize_html.py [input_dir] [output_dir]
"""

import re
import sys
import os
import time
import multiprocessing as mp
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Comment, Doctype, ProcessingInstruction, Tag

from config import RAW_HTML_DIR, SANITIZED_HTML_DIR

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

INPUT_DIR = str(RAW_HTML_DIR)
OUTPUT_DIR = str(SANITIZED_HTML_DIR)

KEEP_ATTRS = {"colspan", "rowspan", "href"}

UNWRAP_TAGS = {
    # Inline XBRL wrappers (present in some 2018 filings)
    "ix:nonnumeric", "ix:nonfraction", "ix:numeric", "ix:fraction",
    "ix:continuation", "ix:header", "ix:references", "ix:resources",
    "ix:exclude",
    # Pure formatting wrappers
    "span", "font",
}

BLOCK_TAGS = {
    "html", "body", "div", "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "table", "thead", "tbody", "tfoot", "tr", "th", "td",
    "ul", "ol", "li", "dl", "dt", "dd",
    "blockquote", "pre", "hr", "br",
    "section", "article", "header", "footer", "main", "nav",
    "a", "b", "strong", "i", "em", "u", "sub", "sup",
}


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def write_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ──────────────────────────────────────────────────────────────
# Step 0 – Pre-parse cleanup
# ──────────────────────────────────────────────────────────────
def pre_parse_cleanup(html: str) -> str:
    """Remove XML declarations, EDGAR document separators, and namespace cruft."""
    html = re.sub(r"<\?xml[^?]*\?>", "", html, flags=re.IGNORECASE)
    # Remove EDGAR document type/header tags
    html = re.sub(r"<DOCUMENT>.*?<TYPE>.*?\n", "", html, flags=re.IGNORECASE)
    html = re.sub(r"</DOCUMENT>", "", html, flags=re.IGNORECASE)
    html = re.sub(r"<SEC-DOCUMENT>.*?</SEC-HEADER>", "", html, flags=re.DOTALL | re.IGNORECASE)
    return html


# ──────────────────────────────────────────────────────────────
# Step 1 – Remove <script> and <style> blocks
# ──────────────────────────────────────────────────────────────
def remove_script_and_style(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(["style", "script", "noscript"]):
        tag.decompose()


# ──────────────────────────────────────────────────────────────
# Step 2 – Remove display:none elements
# ──────────────────────────────────────────────────────────────
def remove_hidden_elements(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        tag.decompose()


# ──────────────────────────────────────────────────────────────
# Step 3 – Unwrap XBRL + formatting-only tags
# ──────────────────────────────────────────────────────────────
def unwrap_xbrl_and_formatting(soup: BeautifulSoup) -> None:
    changed = True
    while changed:
        changed = False
        for tag in soup.find_all():
            if tag.name and tag.name.lower() in UNWRAP_TAGS:
                tag.unwrap()
                changed = True


# ──────────────────────────────────────────────────────────────
# Step 4 – Strip attributes
# ──────────────────────────────────────────────────────────────
def strip_attributes(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(True):
        attrs_to_keep = {}
        for attr in KEEP_ATTRS:
            if attr in tag.attrs:
                attrs_to_keep[attr] = tag.attrs[attr]
        tag.attrs = attrs_to_keep


# ──────────────────────────────────────────────────────────────
# Step 5 – Remove images (base64 and linked)
# ──────────────────────────────────────────────────────────────
def remove_images(soup: BeautifulSoup) -> None:
    for tag in soup.find_all("img"):
        tag.decompose()


# ──────────────────────────────────────────────────────────────
# Step 6 – Purge empty elements
# ──────────────────────────────────────────────────────────────
def is_effectively_empty(tag) -> bool:
    if isinstance(tag, NavigableString):
        return False
    if tag.name in ("br", "hr", "img", "input"):
        return False
    text = tag.get_text(strip=True)
    text = text.replace("\u00a0", "").strip()
    return len(text) == 0


def purge_empty_elements(soup: BeautifulSoup) -> None:
    changed = True
    while changed:
        changed = False
        for tag in soup.find_all(True):
            if is_effectively_empty(tag):
                tag.decompose()
                changed = True


# ──────────────────────────────────────────────────────────────
# Step 7 – Remove ALL tables
# ──────────────────────────────────────────────────────────────
# Financial tables will be reconstructed from parsed XBRL data
# (see recreate_filings.py), so we strip them entirely here.

def remove_all_tables(soup: BeautifulSoup) -> None:
    """Remove every <table> element from the document."""
    for table in soup.find_all("table"):
        table.decompose()


# ──────────────────────────────────────────────────────────────
# Step 8 – Flatten single-child nesting
# ──────────────────────────────────────────────────────────────
def flatten_single_child(soup: BeautifulSoup) -> None:
    changed = True
    while changed:
        changed = False
        for tag in soup.find_all(True):
            children = [c for c in tag.children
                        if not (isinstance(c, NavigableString) and c.strip() == "")]
            if len(children) == 1 and not isinstance(children[0], NavigableString):
                child = children[0]
                if tag.name == child.name or tag.name in ("div", "body"):
                    tag.unwrap()
                    changed = True


# ──────────────────────────────────────────────────────────────
# Step 9 – Normalize entities & whitespace
# ──────────────────────────────────────────────────────────────
def normalize_text(html: str) -> str:
    html = html.replace("\u00a0", " ")
    html = re.sub(r"&#160;", " ", html)
    html = re.sub(r"&nbsp;", " ", html, flags=re.IGNORECASE)
    lines = html.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.lstrip(" \t")
        indent = line[:len(line) - len(stripped)]
        stripped = re.sub(r"[ \t]+", " ", stripped)
        cleaned.append(indent + stripped)
    html = "\n".join(cleaned)
    html = re.sub(r"\n{3,}", "\n\n", html)
    html = re.sub(r"\n[ \t]+\n", "\n\n", html)
    return html.strip()


# ──────────────────────────────────────────────────────────────
# Step 10 – Extract body / remove comments
# ──────────────────────────────────────────────────────────────
def extract_body(soup: BeautifulSoup) -> BeautifulSoup:
    body = soup.find("body")
    if body:
        return BeautifulSoup(str(body), "lxml")
    return soup


def remove_comments(soup: BeautifulSoup) -> None:
    for comment in soup.find_all(
        string=lambda t: isinstance(t, (Comment, Doctype, ProcessingInstruction))
    ):
        comment.extract()


# ──────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────
def sanitize(input_path: str) -> str:
    raw = read_file(input_path)
    print(f"Original size: {len(raw):,} chars")

    raw = pre_parse_cleanup(raw)
    soup = BeautifulSoup(raw, "lxml")

    remove_comments(soup)
    remove_script_and_style(soup)
    remove_hidden_elements(soup)
    remove_images(soup)
    unwrap_xbrl_and_formatting(soup)
    strip_attributes(soup)
    purge_empty_elements(soup)
    soup = extract_body(soup)
    remove_all_tables(soup)
    flatten_single_child(soup)
    purge_empty_elements(soup)

    result = soup.prettify()
    result = normalize_text(result)

    print(f"Sanitized size: {len(result):,} chars")
    print(f"Reduction: {100 * (1 - len(result) / max(len(raw), 1)):.1f}%")
    return result


def _process_one(args: tuple) -> tuple[str, bool, str]:
    """Worker function for multiprocessing. Returns (filename, success, message)."""
    html_path, output_dir = args
    output_path = Path(output_dir) / Path(html_path).name
    if output_path.exists():
        return (Path(html_path).name, True, "skipped (exists)")
    try:
        result = sanitize(html_path)
        write_file(str(output_path), result)
        return (Path(html_path).name, True, "ok")
    except Exception as e:
        return (Path(html_path).name, False, str(e))


def main():
    input_dir = INPUT_DIR
    output_dir = OUTPUT_DIR
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    if not os.path.exists(input_dir):
        print(f"Error: directory not found: {input_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    html_files = sorted(Path(input_dir).glob("*.html"))
    print(f"Found {len(html_files)} HTML files in {input_dir}")
    print(f"Output directory: {output_dir}")

    num_workers = max(1, mp.cpu_count() - 1)
    print(f"Using {num_workers} worker processes\n")

    tasks = [(str(p), output_dir) for p in html_files]
    t0 = time.time()
    done = 0
    errors = 0

    with mp.Pool(num_workers) as pool:
        for filename, success, msg in pool.imap_unordered(_process_one, tasks):
            done += 1
            if not success:
                errors += 1
                print(f"  [{done}/{len(tasks)}] x {filename}: {msg}")
            elif msg != "skipped (exists)":
                print(f"  [{done}/{len(tasks)}] {filename}")

    elapsed = time.time() - t0
    print(f"\nDone! {done} files in {elapsed:.1f}s ({errors} errors).")


if __name__ == "__main__":
    main()
