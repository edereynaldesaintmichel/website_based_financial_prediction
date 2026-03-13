"""
Analyze number contexts in raw 10-K markdown files.

Finds all numbers that are NOT preceded by $ or followed by %,
and collects the word before and after each match. Outputs a
report with all examples and frequency tables for prefixes/suffixes.

Usage:
    python analyze_number_contexts.py [input_dir] [output_file]
"""

import re
import sys
from pathlib import Path
from collections import Counter

from config import LLM_SANITIZED_DIR

INPUT_DIR = str(LLM_SANITIZED_DIR)
OUTPUT_FILE = "number_context_report.txt"

# Matches any "number-like" token: integers, decimals, comma-formatted
NUMBER_RE = re.compile(
    r'(?P<num>'
        r'\d[\d,]*(?:\.\d+)?'
    r')'
)

# What counts as "word before" and "word after"
WORD_BEFORE_RE = re.compile(r'(\S+)\s*$')
WORD_AFTER_RE = re.compile(r'^\s*(\S+)')


def analyze_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    results = []

    for m in NUMBER_RE.finditer(text):
        raw = m.group("num")
        start, end = m.start(), m.end()

        # Skip if preceded by $ (look back for it)
        pre_char_idx = start - 1
        while pre_char_idx >= 0 and text[pre_char_idx] in ' \t':
            pre_char_idx -= 1
        if pre_char_idx >= 0 and text[pre_char_idx] == '$':
            continue

        # Skip if followed by %
        post_char_idx = end
        while post_char_idx < len(text) and text[post_char_idx] in ' \t':
            post_char_idx += 1
        if post_char_idx < len(text) and text[post_char_idx] == '%':
            continue

        # Extract word before
        before_text = text[max(0, start - 50):start]
        bm = WORD_BEFORE_RE.search(before_text)
        word_before = bm.group(1) if bm else ""

        # Extract word after
        after_text = text[end:end + 50]
        am = WORD_AFTER_RE.search(after_text)
        word_after = am.group(1) if am else ""

        results.append({
            "number": raw,
            "word_before": word_before,
            "word_after": word_after,
        })

    return results


def main():
    input_dir = INPUT_DIR
    output_file = OUTPUT_FILE
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    input_path = Path(input_dir)
    md_files = sorted(input_path.glob("*.md"))
    print(f"Found {len(md_files)} files in {input_dir}")

    all_results = []
    prefix_counter = Counter()
    suffix_counter = Counter()

    for i, f in enumerate(md_files, 1):
        file_results = analyze_file(f)
        for r in file_results:
            prefix_counter[r["word_before"]] += 1
            suffix_counter[r["word_after"]] += 1
        all_results.extend([(f.name, r) for r in file_results])
        print(f"[{i}/{len(md_files)}] {f.name}: {len(file_results)} numbers found")

    print(f"\nTotal numbers found (excl. $ and %): {len(all_results)}")
    print(f"Writing report to {output_file}")

    with open(output_file, "w", encoding="utf-8") as out:
        # --- Section 1: Prefix frequencies ---
        out.write("=" * 80 + "\n")
        out.write("PREFIX FREQUENCIES (word immediately before the number)\n")
        out.write("=" * 80 + "\n\n")
        for word, count in prefix_counter.most_common():
            out.write(f"  {count:>8}  {word!r}\n")

        # --- Section 2: Suffix frequencies ---
        out.write("\n" + "=" * 80 + "\n")
        out.write("SUFFIX FREQUENCIES (word immediately after the number)\n")
        out.write("=" * 80 + "\n\n")
        for word, count in suffix_counter.most_common():
            out.write(f"  {count:>8}  {word!r}\n")

        # --- Section 3: All matches with context ---
        out.write("\n" + "=" * 80 + "\n")
        out.write("ALL MATCHES: [prefix] NUMBER [suffix]\n")
        out.write("=" * 80 + "\n\n")
        for filename, r in all_results:
            out.write(
                f"  {filename}:  "
                f"{r['word_before']!r:>30}  |  "
                f"{r['number']:<20}  |  "
                f"{r['word_after']!r}\n"
            )

    print("Done!")


if __name__ == "__main__":
    main()
