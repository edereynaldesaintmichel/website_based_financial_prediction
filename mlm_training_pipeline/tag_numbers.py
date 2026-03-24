"""
Tag numbers in Wikipedia text for MLM training.

Tags every number (integers, decimals, comma-formatted).
Does not capture signs (leading minus/plus).

Usage (standalone):
    python -m wikipedia_pipeline.tag_numbers < input.txt
"""

import re

NUMBER_RE = re.compile(
    r"""(?<![.\d"'])"""            # not preceded by digit, dot, or quote
    r'(?P<num>'
        r'\d{1,3}(?:,\d{3})+'   # comma-formatted (1,234 or 12,345,678)
        r'(?:\.\d+)?'            # optional decimal
    r'|'
        r'\d+'                   # plain integer
        r'(?:\.\d+)?'            # optional decimal
    r')'
    r'(?!\d|\.\d)'               # not followed by digit or decimal point+digit
    r"""(?!["'])"""               # not followed by quote
)


def normalize_number(raw: str) -> str:
    clean = raw.replace(",", "").strip()
    try:
        return str(int(clean))
    except ValueError:
        pass
    try:
        return f"{float(clean):.10f}".rstrip("0").rstrip(".")
    except ValueError:
        return raw


def strip_number_tags(text: str) -> str:
    return text.replace("<number>", "").replace("</number>", "")


def tag_numbers_in_text(text: str) -> str:
    text = strip_number_tags(text)
    def replace(m):
        return f"<number>{normalize_number(m.group('num'))}</number>"
    return NUMBER_RE.sub(replace, text)


if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Tag numbers in text.")
    parser.add_argument("input", nargs="?", help="Input file (default: stdin)")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    args = parser.parse_args()

    text = open(args.input).read() if args.input else sys.stdin.read()
    tagged = tag_numbers_in_text(text)

    if args.output:
        with open(args.output, "w") as f:
            f.write(tagged)
    else:
        print(tagged)

    print(f"--- {tagged.count('<number>')} numbers tagged ---", file=sys.stderr)
