#!/usr/bin/env python3
"""
Phase 3: Recombine markdown files with OCR'd table results.

Takes markdown files (with GLMOCRTABLE{i}GLMOCRTABLE placeholders) and
a JSONL file of OCR'd tables ({"stem": "...", "html": "..."}),
splices them into final markdown.

Usage:
    python recombine.py <markdown_dir> <tables_ocred.jsonl> [--output DIR]
"""

import argparse
import json
import re
import sys
from pathlib import Path

from tqdm import tqdm

PLACEHOLDER_RE = re.compile(r"GLMOCRTABLE(\d+)GLMOCRTABLE")


def load_tables(jsonl_path: Path) -> dict[str, str]:
    """Load JSONL into a {stem: html} dict."""
    tables = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                tables[entry["stem"]] = entry["html"]
    return tables


def recombine_file(md_path: Path, tables: dict[str, str]) -> tuple[str, int]:
    """Replace placeholders with OCR'd table content. Returns (result, n_missing)."""
    stem = md_path.stem
    markdown = md_path.read_text(encoding="utf-8")
    n_missing = 0

    def replace_placeholder(match):
        nonlocal n_missing
        idx = match.group(1)
        key = f"{stem}__t{idx}"
        if key in tables:
            return f"\n\n{tables[key]}\n\n"
        else:
            n_missing += 1
            return f"\n\n<!-- Missing table {idx} -->\n\n"

    result = PLACEHOLDER_RE.sub(replace_placeholder, markdown)
    result = re.sub(r"\n{4,}", "\n\n\n", result)
    return result, n_missing


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Recombine markdown + OCR'd tables"
    )
    parser.add_argument("markdown", help="Directory of markdown files with placeholders")
    parser.add_argument("tables", help="JSONL file of OCR'd tables ({stem, html} per line)")
    parser.add_argument("--output", help="Output directory (default: {markdown}_final)")
    args = parser.parse_args()

    md_dir = Path(args.markdown)
    tables_path = Path(args.tables)

    if not md_dir.is_dir():
        print(f"Error: {md_dir} is not a directory")
        sys.exit(1)
    if not tables_path.is_file():
        print(f"Error: {tables_path} is not a file")
        sys.exit(1)

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = md_dir.parent / f"{md_dir.name}_final"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tables from {tables_path}...")
    tables = load_tables(tables_path)
    print(f"Loaded {len(tables)} tables")

    md_files = sorted(md_dir.glob("*.md"))

    print(f"Markdown: {md_dir} ({len(md_files)} files)")
    print(f"Output:   {output_dir}")

    n_done = 0
    total_missing = 0

    for md_path in tqdm(md_files, desc="Recombine", unit="file"):
        result, n_missing = recombine_file(md_path, tables)
        (output_dir / md_path.name).write_text(result, encoding="utf-8")
        total_missing += n_missing
        n_done += 1

    print(f"\nDone! {n_done} files recombined → {output_dir}")
    if total_missing:
        print(f"  {total_missing} tables had no OCR result (marked <!-- Missing table -->)")


if __name__ == "__main__":
    main()
