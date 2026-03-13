"""
Tag Numbers in Doubly-Sanitized HTML Financial Filings.

Reads each doubly-sanitized HTML file and wraps financial numbers
in <number> tags with normalized formatting:
  - Commas are removed:        1,000  →  <number>1000</number>
  - Brackets mean negative:    (532)  →  <number>-532</number>
  - Scientific notation:       1.5e3  →  <number>1500</number>  (not expected, but handled)
  - Years are excluded:        2017 stays as 2017 (no tag)
  - Small pipe-cell numbers:   | 42 | →  | <number>42</number> |  (markdown table context)

A "year" is defined as a 4-digit integer in the range 1900–2099 that does
NOT contain commas (i.e. it was written plainly as e.g. "2017", "1999").
Numbers ≥1000 without commas that match the year range are treated as years.

Small numbers (1–4 digits) are only tagged when they appear inside a markdown
table cell, i.e. surrounded by pipe characters (|), to avoid false positives
such as page numbers, footnote indices, or ordinary text numbers.

Usage:
    python tag_numbers.py
"""

import re
import os
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

INPUT_DIR = "/Users/eloireynal/Downloads/html_50plus_employees_gemini_sanitized"
OUTPUT_DIR = "/Users/eloireynal/Downloads/html_50plus_employees_tagged"

# File extension produced by the Gemini step
FILE_EXT = "*.md"


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def write_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ──────────────────────────────────────────────────────────────
# Year detection
# ──────────────────────────────────────────────────────────────

def is_year(raw: str) -> bool:
    """
    Determine if a numeric string is a year.
    Years are 4-digit numbers in [1900, 2099] with NO commas.
    """
    # If the raw text contained commas, it's not a year (it's a formatted number)
    if "," in raw:
        return False
    # Strip any whitespace and try to parse
    clean = raw.replace(" ", "")
    try:
        val = int(clean)
        return 1900 <= val <= 2099
    except ValueError:
        return False


# ──────────────────────────────────────────────────────────────
# Number normalization
# ──────────────────────────────────────────────────────────────

def normalize_number(raw: str, is_negative: bool = False) -> str:
    """
    Normalize a matched number string:
      - Remove commas
      - Parse as int or float
      - No scientific notation
      - Prepend minus if negative
    """
    # Remove commas and whitespace
    clean = raw.replace(",", "").replace(" ", "").strip()

    # Try integer first
    try:
        val = int(clean)
        result = str(val)
        if is_negative:
            result = f"-{result}"
        return result
    except ValueError:
        pass

    # Try float
    try:
        val = float(clean)
        # Format without scientific notation
        # Use up to 10 decimal places, strip trailing zeros
        result = f"{val:.10f}".rstrip("0").rstrip(".")
        if is_negative:
            result = f"-{result}"
        return result
    except ValueError:
        return raw


# ──────────────────────────────────────────────────────────────
# Regex pattern — single combined pattern for one-pass matching
# ──────────────────────────────────────────────────────────────

# Combined pattern that matches (in order of priority):
#   Group "bracket":   (1,234) or (532) — bracketed negatives
#   Group "comma":     1,234 or 12,345,678.90 — comma-separated numbers
#   Group "large":     10000+ — plain 5+ digit integers
COMBINED_RE = re.compile(
    r'(?P<bracket>'
        r'\('
        r'\s*'
        r'(?:\d{1,3}(?:,\d{3})*|\d+)'  # integer part with optional commas
        r'(?:\.\d+)?'                    # optional decimal part
        r'\s*'
        r'\)'
    r')'
    r'|'
    r'(?<![.\d])'                        # not preceded by digit or dot
    r'(?P<comma>'
        r'\d{1,3}'
        r'(?:,\d{3})+'                  # at least one comma group
        r'(?:\.\d+)?'                    # optional decimal part
    r')'
    r'(?![,\d])'                         # not followed by comma or digit
    r'|'
    r'(?<![.,\d])'                       # not preceded by digit, dot, or comma
    r'(?P<large>\d{5,})'                # 5+ digit integers
    r'(?!\d)'                            # not followed by digit
)

# Separate pattern for small numbers inside markdown table cells.
# Matches a full pipe-to-pipe cell chunk so we can inspect and replace
# small numbers (1–4 digits) within it without a variable-width lookbehind.
# Group "cell" captures the content between two pipes (including surrounding spaces).
PIPE_CELL_RE = re.compile(
    r'(?<=\|)'          # immediately after a pipe  (fixed-width lookbehind — OK)
    r'(?P<cell>[^|\n]*)'  # capture cell content up to next pipe or newline
    r'(?=\|)'           # immediately before a pipe (lookahead — OK)
)

# Matches a cell whose ENTIRE content (ignoring surrounding whitespace and
# optional markdown bold markers **) is a small number (1–4 digits, optional decimal).
# The outer groups preserve the leading/trailing whitespace and bold markers so
# the replacement can put them back unchanged.
_CELL_PURE_NUM_RE = re.compile(
    r'^(\s*\*{0,2}\s*)'       # leading whitespace + optional ** bold marker
    r'(-?\d{1,4}(?:\.\d+)?)' # the number itself
    r'(\s*\*{0,2}\s*)$'       # trailing optional ** bold marker + whitespace
)

# Pattern for bold-formatted numbers: **[£$€]?NUMBER[%]**
# Handles optional currency prefix (£, $, €) and optional percent suffix.
# Applied after COMBINED_RE so it only catches numbers the main pass missed
# (i.e. small decimals like 5.8 or 45.36 that lack commas and are < 5 digits).
# Numbers already inside <number> tags won't match because the tag characters
# break the digit sequence required by the number group.
BOLD_NUM_RE = re.compile(
    r'\*\*'
    r'(\s*[£$€]?\s*)'               # group 1: optional currency + surrounding spaces
    r'(-?\d[\d,]*(?:\.\d+)?)'       # group 2: the number (int or decimal, optional commas)
    r'(\s*%?\s*)'                   # group 3: optional percent sign + surrounding spaces
    r'\*\*'
)


def tag_numbers_in_text(text: str) -> str:
    """
    Process text and wrap numbers in <number> tags.
    Handles bracket-negatives, comma-separated numbers, and large plain numbers.
    Avoids tagging years.
    Uses a single-pass combined regex to prevent double-tagging.
    """

    # We need to be careful not to modify text inside HTML tags.
    # Strategy: split text into HTML tags and text segments, only process text segments.

    # Split into segments: HTML tags vs text content
    segments = re.split(r'(<[^>]+>)', text)

    result = []
    for segment in segments:
        if segment.startswith('<') and segment.endswith('>'):
            # HTML tag — pass through unchanged
            result.append(segment)
        else:
            # Text content — process numbers
            processed = _process_text_segment(segment)
            result.append(processed)

    combined = ''.join(result)
    # Second pass: tag numbers inside bold markers that the main regex missed
    # (e.g. small decimals like **5.8 %** or **45.36 %**).
    return BOLD_NUM_RE.sub(_replace_bold_num, combined)


def _replace_number(match) -> str:
    """Replacement function for the combined regex."""
    if match.group("bracket"):
        # Bracketed (negative) number
        full = match.group("bracket")
        # Extract the number inside brackets
        inner = full[1:-1].strip()  # Remove ( and )
        if is_year(inner):
            return full  # Don't tag years
        normalized = normalize_number(inner, is_negative=True)
        return f"<number>{normalized}</number>"

    elif match.group("comma"):
        # Comma-separated number
        raw = match.group("comma")
        # Has commas, so can't be a year
        normalized = normalize_number(raw)
        return f"<number>{normalized}</number>"

    elif match.group("large"):
        # Large plain integer (5+ digits)
        raw = match.group("large")
        normalized = normalize_number(raw)
        return f"<number>{normalized}</number>"

    return match.group(0)  # fallback: no change


def _tag_small_in_cell(cell_text: str) -> str:
    """
    Tag a table cell ONLY if its entire content (ignoring surrounding whitespace
    and optional markdown bold markers **) is a small number (1–4 digits,
    optional decimal part). Mixed cells like " 30th April 2017 " are left as-is.
    Years (1900–2099) are never tagged.
    """
    m = _CELL_PURE_NUM_RE.match(cell_text)
    if not m:
        return cell_text  # cell contains more than just a number — leave it alone
    prefix, raw, suffix = m.group(1), m.group(2), m.group(3)
    if is_year(raw):
        return cell_text  # don't tag years
    normalized = normalize_number(raw)
    return f"{prefix}<number>{normalized}</number>{suffix}"


def _apply_pipe_cell_tagging(line: str) -> str:
    """
    For a markdown table line (contains at least one pipe), tag small numbers
    that appear inside cells without disturbing the pipe structure.
    """
    def replace_cell(m):
        return _tag_small_in_cell(m.group("cell"))

    return PIPE_CELL_RE.sub(replace_cell, line)


def _replace_bold_num(m: re.Match) -> str:
    """Replacement for BOLD_NUM_RE: tag the number inside bold markers.

    Years (1900–2099 without commas) are left untagged.
    """
    prefix = m.group(1)   # currency / leading spaces
    raw = m.group(2)       # the bare number string
    suffix = m.group(3)    # percent / trailing spaces
    if is_year(raw):
        return m.group(0)
    normalized = normalize_number(raw)
    return f'**{prefix}<number>{normalized}</number>{suffix}**'


def _process_text_segment(text: str) -> str:
    """Process a text segment (not inside HTML tags) to tag numbers.

    For lines that look like markdown table rows (contain '|'), small numbers
    inside cells are tagged first via PIPE_CELL_RE before the main COMBINED_RE
    handles commas, brackets, and large numbers.
    """
    lines = text.split('\n')
    processed_lines = []
    for line in lines:
        if '|' in line:
            # First pass: tag small numbers inside table cells
            line = _apply_pipe_cell_tagging(line)
        # Second pass: tag bracket-negatives, comma-numbers, and large integers
        processed_lines.append(COMBINED_RE.sub(_replace_number, line))
    return '\n'.join(processed_lines)


# ──────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────

def process_file(input_path: str, output_path: str) -> dict:
    """Process a single file: tag numbers and save."""
    text = read_file(input_path)
    result = tag_numbers_in_text(text)

    # Count the number of tags added
    num_tags = result.count("<number>")

    write_file(output_path, result)
    return {"tags": num_tags}


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

    html_files = sorted(Path(input_dir).glob(FILE_EXT))
    print(f"📂 Found {len(html_files)} files in {input_dir}")
    print(f"📂 Output directory: {output_dir}\n")

    total_tags = 0
    for i, html_path in enumerate(html_files, 1):
        output_path = Path(output_dir) / html_path.name
        try:
            stats = process_file(str(html_path), str(output_path))
            total_tags += stats["tags"]
            print(f"[{i}/{len(html_files)}] {html_path.name}: {stats['tags']} numbers tagged")
        except Exception as e:
            print(f"[{i}/{len(html_files)}] {html_path.name}: ✗ Error: {e}")

    print(f"\n✅ Done! Processed {len(html_files)} files.")
    print(f"📊 Total numbers tagged: {total_tags:,}")


if __name__ == "__main__":
    main()
