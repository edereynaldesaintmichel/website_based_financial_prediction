"""
Sanitize iXBRL financial HTML files for LLM consumption.

Techniques applied:
1. Remove all <style> blocks and inline style attributes.
2. Remove elements with display:none (inline or via known CSS classes).
3. Strip (almost) every attribute from every element.
4. Unwrap inline XBRL tags (<ix:*>) keeping their text content.
5. Purge empty / decoration-only elements (horizontal lines, spacers).
6. Flatten degenerate tables (single-cell tables → <p>; layout-only tables).
7. Flatten single-child nesting (reduce DOM depth).
8. Normalize HTML entities (&#160; → space, &amp; → &, etc.).
9. Collapse excessive whitespace.
10. Strip <head>, XML declarations, and namespace cruft.
"""

import re
import sys
import os
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Comment, Doctype, ProcessingInstruction, Tag

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

INPUT_DIR = "/Users/eloireynal/Downloads/html_50plus_employees"
OUTPUT_DIR = "/Users/eloireynal/Downloads/html_50plus_employees_sanitized"

# Attributes we *keep* (everything else is stripped)
KEEP_ATTRS = {"colspan", "rowspan", "href"}

# Tags whose open/close wrappers we remove, keeping inner content
UNWRAP_TAGS = {
    # Inline XBRL wrappers
    "ix:nonnumeric", "ix:nonfraction", "ix:numeric", "ix:fraction",
    "ix:continuation", "ix:header", "ix:references", "ix:resources",
    "ix:exclude",
    # Pure formatting wrappers that add nothing semantic
    "span", "font",
}

# Tags that are structural / semantic and should survive flattening
BLOCK_TAGS = {
    "html", "body", "div", "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "table", "thead", "tbody", "tfoot", "tr", "th", "td",
    "ul", "ol", "li", "dl", "dt", "dd",
    "blockquote", "pre", "hr", "br",
    "section", "article", "header", "footer", "main", "nav",
    "a", "b", "strong", "i", "em", "u", "sub", "sup",
}

# CSS classes known to produce display:none from the stylesheet
DISPLAY_NONE_CLASSES = {"tooltipRTF"}


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
    """Remove XML declaration and fix namespace prefixes so
    BeautifulSoup's html.parser can handle them."""
    # Strip <?xml ...?>
    html = re.sub(r"<\?xml[^?]*\?>", "", html, flags=re.IGNORECASE)
    return html


# ──────────────────────────────────────────────────────────────
# Step 1 – Remove <style> blocks
# ──────────────────────────────────────────────────────────────
def remove_style_blocks(soup: BeautifulSoup) -> None:
    for tag in soup.find_all("style"):
        tag.decompose()


# ──────────────────────────────────────────────────────────────
# Step 2 – Remove display:none elements
# ──────────────────────────────────────────────────────────────
def remove_hidden_elements(soup: BeautifulSoup) -> None:
    # Inline style display:none
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        tag.decompose()

    # Known hidden classes from stylesheet analysis
    for cls in DISPLAY_NONE_CLASSES:
        for tag in soup.find_all(class_=cls):
            tag.decompose()


# ──────────────────────────────────────────────────────────────
# Step 3 – Unwrap XBRL + formatting-only tags
# ──────────────────────────────────────────────────────────────
def unwrap_xbrl_and_formatting(soup: BeautifulSoup) -> None:
    """Replace <ix:nonNumeric ...>text</ix:nonNumeric> → text, etc."""
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
# Step 5 – Purge empty elements
# ──────────────────────────────────────────────────────────────
def is_effectively_empty(tag) -> bool:
    """Return True if a tag contains no meaningful text."""
    if isinstance(tag, NavigableString):
        return False
    # Self-closing tags that are meaningful
    if tag.name in ("br", "hr", "img", "input"):
        return False
    text = tag.get_text(strip=True)
    # Also treat non-breaking-space-only as empty
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
# Step 6 – Flatten degenerate / layout-only tables
# ──────────────────────────────────────────────────────────────
def _meaningful_children(tag: Tag):
    """Return non-whitespace children of a tag."""
    return [c for c in tag.children
            if not (isinstance(c, NavigableString) and c.strip() == "")]


def _is_degenerate_table(table: Tag) -> bool:
    """
    A table is 'degenerate' (layout-only, not real data) when every row
    has exactly one cell (colspan=1 or absent) and there are no <th> cells.
    Real data tables have multiple columns or explicit headers.
    """
    rows = [c for c in table.descendants if isinstance(c, Tag) and c.name == "tr"]
    if not rows:
        return False
    for row in rows:
        cells = [c for c in row.children
                 if isinstance(c, Tag) and c.name in ("td", "th")]
        # If any row has a th → it's a real header → keep table
        if any(c.name == "th" for c in cells):
            return False
        # If any row has colspan > 1 or more than one cell → real table
        for cell in cells:
            try:
                cs = int(cell.get("colspan", 1))
            except (ValueError, TypeError):
                cs = 1
            if cs > 1:
                return False
        if len(cells) > 1:
            return False
    return True


def flatten_degenerate_tables(soup: BeautifulSoup) -> None:
    """
    Replace degenerate tables (single-column, no headers) with their
    cell contents wrapped in <p> tags.
    """
    changed = True
    while changed:
        changed = False
        for table in soup.find_all("table"):
            if not _is_degenerate_table(table):
                continue

            parent = table.parent
            if parent is None:
                continue

            # Gather replacement nodes (in DOM order via find_all("td"))
            replacement_nodes = []
            for td in table.find_all("td"):
                mc = _meaningful_children(td)
                if not mc:
                    continue
                block_only = all(
                    isinstance(c, Tag) and c.name in BLOCK_TAGS
                    for c in mc
                )
                if block_only:
                    for child in mc:
                        replacement_nodes.append(child.extract())
                else:
                    p = soup.new_tag("p")
                    for child in list(td.children):
                        p.append(child.extract())
                    replacement_nodes.append(p)

            # Replace the table in-place using insert_before
            for node in replacement_nodes:
                table.insert_before(node)
            table.decompose()
            changed = True


# ──────────────────────────────────────────────────────────────
# Step 7 – Flatten single-child nesting
# ──────────────────────────────────────────────────────────────
def flatten_single_child(soup: BeautifulSoup) -> None:
    """If a tag has exactly one child and that child is another tag,
    remove the outer wrapper (keeping the inner tag)."""
    changed = True
    while changed:
        changed = False
        for tag in soup.find_all(True):
            children = [c for c in tag.children
                        if not (isinstance(c, NavigableString) and c.strip() == "")]
            if len(children) == 1 and not isinstance(children[0], NavigableString):
                child = children[0]
                # Don't flatten if they are different meaningful structural tags
                # (e.g., don't merge a <td> into its parent <tr>).
                # But do flatten div>div, div>p, p>p, etc.
                if tag.name == child.name or tag.name in ("div", "body"):
                    tag.unwrap()
                    changed = True


# ──────────────────────────────────────────────────────────────
# Step 8 – Normalize entities & whitespace
# ──────────────────────────────────────────────────────────────
def normalize_text(html: str) -> str:
    # Convert leftover &#160; and &nbsp; to regular spaces
    html = html.replace("\u00a0", " ")
    html = re.sub(r"&#160;", " ", html)
    html = re.sub(r"&nbsp;", " ", html, flags=re.IGNORECASE)

    # Collapse inline runs of whitespace but preserve leading indentation
    # Process line-by-line to keep prettify's indentation intact
    lines = html.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.lstrip(" \t")
        indent = line[:len(line) - len(stripped)]
        # Collapse multiple spaces/tabs within the content part only
        stripped = re.sub(r"[ \t]+", " ", stripped)
        cleaned.append(indent + stripped)
    html = "\n".join(cleaned)
    # Collapse 3+ consecutive newlines into 2
    html = re.sub(r"\n{3,}", "\n\n", html)
    # Remove whitespace-only lines
    html = re.sub(r"\n[ \t]+\n", "\n\n", html)
    return html.strip()


# ──────────────────────────────────────────────────────────────
# Step 9 – Remove non-body content
# ──────────────────────────────────────────────────────────────
def extract_body(soup: BeautifulSoup) -> BeautifulSoup:
    """Keep only the <body> subtree."""
    body = soup.find("body")
    if body:
        # Re-parse just the body content to get a clean soup
        return BeautifulSoup(str(body), "html.parser")
    return soup


# ──────────────────────────────────────────────────────────────
# Step 10 – Remove comments and processing instructions
# ──────────────────────────────────────────────────────────────
def remove_comments(soup: BeautifulSoup) -> None:
    for comment in soup.find_all(string=lambda t: isinstance(t, (Comment, Doctype, ProcessingInstruction))):
        comment.extract()


# ──────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────
def sanitize(input_path: str) -> str:
    raw = read_file(input_path)
    print(f"Original size: {len(raw):,} chars")

    # Pre-parse
    raw = pre_parse_cleanup(raw)

    # Parse
    soup = BeautifulSoup(raw, "html.parser")

    # Pipeline (order matters)
    remove_comments(soup)
    remove_style_blocks(soup)
    remove_hidden_elements(soup)
    unwrap_xbrl_and_formatting(soup)
    strip_attributes(soup)
    purge_empty_elements(soup)
    soup = extract_body(soup)

    # Core new step: collapse single-cell tables into <p> tags
    flatten_degenerate_tables(soup)

    # Standard single-child flattening (div>div, etc.)
    flatten_single_child(soup)

    # Second pass: flattening may have created new empty wrappers
    purge_empty_elements(soup)

    # Serialize & post-process
    result = soup.prettify()
    result = normalize_text(result)

    print(f"Sanitized size: {len(result):,} chars")
    print(f"Reduction: {100 * (1 - len(result) / len(raw)):.1f}%")
    return result


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

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Collect all HTML files
    html_files = sorted(Path(input_dir).glob("*.html"))
    print(f"Found {len(html_files)} HTML files in {input_dir}")
    print(f"Output directory: {output_dir}\n")

    for i, html_path in enumerate(html_files, 1):
        print(f"\n[{i}/{len(html_files)}] Processing: {html_path.name}")
        try:
            result = sanitize(str(html_path))
            output_path = Path(output_dir) / html_path.name
            write_file(str(output_path), result)
            print(f"  → Saved to: {output_path}")
        except Exception as e:
            print(f"  ✗ Error processing {html_path.name}: {e}")

    print(f"\n✅ Done! Processed {len(html_files)} files.")


if __name__ == "__main__":
    main()
