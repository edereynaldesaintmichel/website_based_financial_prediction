"""
Tag every number in text with <number>...</number>.

Unlike the SEC/Wikipedia taggers, this tags ALL numbers — including dates,
years, page numbers, everything. The only exception is numbers written as
words (e.g. "ten").

Usage (standalone):
    python -m glm_ocr_pipeline.tag_numbers < input.txt
    python -m glm_ocr_pipeline.tag_numbers input.md -o output.md
"""

import re
import sys
from pathlib import Path


# Match any numeric token: integers, decimals, comma-formatted.
# Captures negative numbers (leading minus) too.
NUMBER_RE = re.compile(
    r'(?<![.\d,\w])'          # not preceded by digit, dot, comma, or word char
    r'(?P<neg>-)?'             # optional negative sign
    r'(?P<num>'
        r'\d{1,3}(?:,\d{3})+'   # comma-formatted (1,234 or 12,345,678)
        r'(?:\.\d+)?'           # optional decimal
    r'|'
        r'\d+'                  # plain integer
        r'(?:\.\d+)?'           # optional decimal
    r')'
    r'(?![,\d])'               # not followed by digit or comma
)


def normalize_number(raw: str, negative: bool = False) -> str:
    """Strip commas, normalize to clean numeric string. Avoids float precision issues."""
    clean = raw.replace(",", "").strip()
    sign = "-" if negative else ""

    if "." in clean:
        # Keep exact decimal representation — don't round-trip through float
        integer_part, decimal_part = clean.split(".", 1)
        decimal_part = decimal_part.rstrip("0") or "0"
        if not integer_part:
            integer_part = "0"
        return f"{sign}{int(integer_part)}.{decimal_part}"
    else:
        try:
            return f"{sign}{int(clean)}"
        except ValueError:
            return sign + clean


def _replace_number(match: re.Match) -> str:
    raw = match.group("num")
    neg = match.group("neg") == "-"
    normalized = normalize_number(raw, negative=neg)
    return f"<number>{normalized}</number>"


def tag_numbers_in_text(text: str) -> str:
    """Tag every number in text with <number>...</number>."""
    return NUMBER_RE.sub(_replace_number, text)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tag all numbers in text")
    parser.add_argument("input", nargs="?", help="Input file (default: stdin)")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    args = parser.parse_args()

    if args.input:
        text = Path(args.input).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()

    tagged = tag_numbers_in_text(text)

    if args.output:
        Path(args.output).write_text(tagged, encoding="utf-8")
    else:
        print(tagged)

    count = tagged.count("<number>")
    print(f"--- {count} numbers tagged ---", file=sys.stderr)
