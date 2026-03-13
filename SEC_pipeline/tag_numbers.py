"""
Tag Numbers in LLM-Sanitized SEC 10-K Markdown Files.

Identical logic to the UK pipeline's tag_numbers.py — wraps financial
numbers in <number> tags with normalized formatting. Currency context
is USD rather than GBP, but the tagging logic is currency-agnostic.

Usage:
    python tag_numbers.py [input_dir] [output_dir]
"""

import re
import os
import sys
from pathlib import Path

from config import LLM_SANITIZED_DIR, TAGGED_DIR

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

INPUT_DIR = str(LLM_SANITIZED_DIR)
OUTPUT_DIR = str(TAGGED_DIR)
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
    if "," in raw:
        return False
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
    clean = raw.replace(",", "").replace(" ", "").strip()
    try:
        val = int(clean)
        result = str(val)
        if is_negative:
            result = f"-{result}"
        return result
    except ValueError:
        pass
    try:
        val = float(clean)
        result = f"{val:.10f}".rstrip("0").rstrip(".")
        if is_negative:
            result = f"-{result}"
        return result
    except ValueError:
        return raw


# ──────────────────────────────────────────────────────────────
# Regex patterns
# ──────────────────────────────────────────────────────────────

COMBINED_RE = re.compile(
    # --- percent: number followed by % (e.g. 10%, 3.5%, 1,000%) ---
    r'(?<![.\d])'
    r'(?P<percent>'
        r'\d[\d,]*(?:\.\d+)?'
    r')'
    r'(?=\s*%)'
    r'|'
    # --- dollar: number preceded by $ (e.g. $48.8, $1,234) ---
    r'(?<=\$)'
    r'\s*'
    r'(?P<dollar>'
        r'\d[\d,]*(?:\.\d+)?'
    r')'
    r'(?!\d)'
    r'|'
    # --- bracket: (number) for negatives ---
    r'(?P<bracket>'
        r'\('
        r'\s*'
        r'(?:\d{1,3}(?:,\d{3})*|\d+)'
        r'(?:\.\d+)?'
        r'\s*'
        r'\)'
    r')'
    r'|'
    # --- comma: comma-formatted numbers (e.g. 124,000) ---
    r'(?<![.\d])'
    r'(?P<comma>'
        r'\d{1,3}'
        r'(?:,\d{3})+'
        r'(?:\.\d+)?'
    r')'
    r'(?!,\d{3}|\d)'
    # r'|'
    # --- large: 5+ digit plain numbers ---
    # r'(?<![.,\d])'
    # r'(?P<large>\d{5,})'
    # r'(?!\d)'
)

PIPE_CELL_RE = re.compile(
    r'(?<=\|)'
    r'(?P<cell>[^|\n]*)'
    r'(?=\|)'
)

_CELL_PURE_NUM_RE = re.compile(
    r'^(\s*\*{0,2}\s*)'
    r'(-?\d{1,4}(?:\.\d+)?)'
    r'(\s*\*{0,2}\s*)$'
)

BOLD_NUM_RE = re.compile(
    r'\*\*'
    r'(\s*[£$€]?\s*)'
    r'(-?\d[\d,]*(?:\.\d+)?)'
    r'(\s*%?\s*)'
    r'\*\*'
)


def tag_numbers_in_text(text: str) -> str:
    segments = re.split(r'(<[^>]+>)', text)
    result = []
    inside_number_tag = False
    for segment in segments:
        if segment == '<number>':
            inside_number_tag = True
            result.append(segment)
        elif segment == '</number>':
            inside_number_tag = False
            result.append(segment)
        elif segment.startswith('<') and segment.endswith('>'):
            result.append(segment)
        elif inside_number_tag:
            result.append(segment)
        else:
            processed = _process_text_segment(segment)
            result.append(processed)
    combined = ''.join(result)
    return BOLD_NUM_RE.sub(_replace_bold_num, combined)


def _replace_number(match) -> str:
    if match.group("percent"):
        raw = match.group("percent")
        normalized = normalize_number(raw)
        return f"<number>{normalized}</number>"
    elif match.group("dollar"):
        raw = match.group("dollar")
        normalized = normalize_number(raw)
        return f"<number>{normalized}</number>"
    elif match.group("bracket"):
        full = match.group("bracket")
        inner = full[1:-1].strip()
        if is_year(inner):
            return full
        normalized = normalize_number(inner, is_negative=True)
        return f"<number>{normalized}</number>"
    elif match.group("comma"):
        raw = match.group("comma")
        normalized = normalize_number(raw)
        return f"<number>{normalized}</number>"
    # elif match.group("large"):
    #     raw = match.group("large")
    #     normalized = normalize_number(raw)
        return f"<number>{normalized}</number>"
    return match.group(0)


def _tag_small_in_cell(cell_text: str) -> str:
    m = _CELL_PURE_NUM_RE.match(cell_text)
    if not m:
        return cell_text
    prefix, raw, suffix = m.group(1), m.group(2), m.group(3)
    if is_year(raw):
        return cell_text
    normalized = normalize_number(raw)
    return f"{prefix}<number>{normalized}</number>{suffix}"


def _apply_pipe_cell_tagging(line: str) -> str:
    def replace_cell(m):
        return _tag_small_in_cell(m.group("cell"))
    return PIPE_CELL_RE.sub(replace_cell, line)


def _replace_bold_num(m: re.Match) -> str:
    prefix = m.group(1)
    raw = m.group(2)
    suffix = m.group(3)
    if is_year(raw):
        return m.group(0)
    normalized = normalize_number(raw)
    return f'**{prefix}<number>{normalized}</number>{suffix}**'


def _process_text_segment(text: str) -> str:
    lines = text.split('\n')
    processed_lines = []
    for line in lines:
        if '|' in line:
            line = _apply_pipe_cell_tagging(line)
        processed_lines.append(COMBINED_RE.sub(_replace_number, line))
    return '\n'.join(processed_lines)


# ──────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────

def process_file(input_path: str, output_path: str) -> dict:
    text = read_file(input_path)
    result = tag_numbers_in_text(text)
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

    md_files = sorted(Path(input_dir).glob(FILE_EXT))
    print(f"Found {len(md_files)} files in {input_dir}")
    print(f"Output directory: {output_dir}\n")

    total_tags = 0
    for i, md_path in enumerate(md_files, 1):
        output_path = Path(output_dir) / md_path.name
        try:
            stats = process_file(str(md_path), str(output_path))
            total_tags += stats["tags"]
            print(f"[{i}/{len(md_files)}] {md_path.name}: {stats['tags']} numbers tagged")
        except Exception as e:
            print(f"[{i}/{len(md_files)}] {md_path.name}: x Error: {e}")

    print(f"\nDone! Processed {len(md_files)} files.")
    print(f"Total numbers tagged: {total_tags:,}")


if __name__ == "__main__":
    main()
