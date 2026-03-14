"""
Tag numbers in Wikipedia text for MLM training.

Simpler than the SEC/financial tagger:
- Tags any number that isn't a year (4-digit number without commas, 1000–2099)
- Does NOT treat (number) as negative (that's a financial accounting convention)
- No special handling for markdown tables, bold formatting, or currency symbols

Usage (standalone):
    python -m wikipedia_pipeline.tag_numbers < input.txt
"""

import re


# ──────────────────────────────────────────────────────────────
# Year detection
# ──────────────────────────────────────────────────────────────

def is_year(raw: str) -> bool:
    """A year is a plain 4-digit number in the 1000–2099 range, no commas."""
    if "," in raw:
        return False
    clean = raw.replace(" ", "")
    try:
        val = int(clean)
        return 1000 <= val <= 2099
    except ValueError:
        return False


# ──────────────────────────────────────────────────────────────
# Number normalization
# ──────────────────────────────────────────────────────────────

def normalize_number(raw: str) -> str:
    """Strip commas/spaces, normalize to clean int or float string."""
    clean = raw.replace(",", "").replace(" ", "").strip()
    try:
        val = int(clean)
        return str(val)
    except ValueError:
        pass
    try:
        val = float(clean)
        return f"{val:.10f}".rstrip("0").rstrip(".")
    except ValueError:
        return raw


# ──────────────────────────────────────────────────────────────
# Regex: match any number-like token
# ──────────────────────────────────────────────────────────────

# Matches integers, decimals, and comma-formatted numbers.
# Negative sign is NOT captured (Wikipedia doesn't use accounting brackets,
# and a leading minus could be a dash/hyphen — safer to skip).
NUMBER_RE = re.compile(
    r'(?<![.\d,])'          # not preceded by digit, dot, or comma
    r'(?P<num>'
        r'\d{1,3}(?:,\d{3})+' # comma-formatted (e.g. 1,234 or 12,345,678)
        r'(?:\.\d+)?'          # optional decimal part
    r'|'
        r'\d+'                 # plain integer (e.g. 42, 100000)
        r'(?:\.\d+)?'          # optional decimal part
    r')'
    r'(?![,\d])'             # not followed by digit or comma
)

def tag_numbers_in_text(text: str) -> str:
    """Tag all numbers in text except years."""
    return NUMBER_RE.sub(_replace_number, text)


def _replace_number(match: re.Match) -> str:
    raw = match.group("num")
    if is_year(raw):
        return raw
    normalized = normalize_number(raw)
    return f"<number>{normalized}</number>"


# ──────────────────────────────────────────────────────────────
# CLI for quick testing
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    text = sys.stdin.read()
    tagged = tag_numbers_in_text(text)
    print(tagged)
    print(f"\n--- {tagged.count('<number>')} numbers tagged ---", file=sys.stderr)
