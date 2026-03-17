#!/usr/bin/env python3
"""
Phase 3: Recombine markdown files with OCR'd table results.

Takes markdown files (with GLMOCRTABLE{i}GLMOCRTABLE placeholders) and
OCR'd table files ({stem}__t{i}.html), splices them into final markdown.

Usage:
    python recombine.py <markdown_dir> <tables_cleaned_up_dir> [--output DIR]
"""

import argparse
import re
import sys
from pathlib import Path

from tqdm import tqdm

PLACEHOLDER_RE = re.compile(r"GLMOCRTABLE(\d+)GLMOCRTABLE")


def recombine_file(md_path: Path, tables_dir: Path) -> tuple[str, int]:
    """Replace placeholders with OCR'd table content. Returns (result, n_missing)."""
    stem = md_path.stem
    markdown = md_path.read_text(encoding="utf-8")
    n_missing = 0

    def replace_placeholder(match):
        nonlocal n_missing
        idx = match.group(1)
        table_file = tables_dir / f"{stem}__t{idx}.html"
        if table_file.exists():
            return f"\n\n{table_file.read_text(encoding='utf-8')}\n\n"
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
    parser.add_argument("tables", help="Directory of OCR'd table .html files")
    parser.add_argument("--output", help="Output directory (default: {markdown}_final)")
    args = parser.parse_args()

    md_dir = Path(args.markdown)
    tables_dir = Path(args.tables)

    if not md_dir.is_dir():
        print(f"Error: {md_dir} is not a directory")
        sys.exit(1)
    if not tables_dir.is_dir():
        print(f"Error: {tables_dir} is not a directory")
        sys.exit(1)

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = md_dir.parent / f"{md_dir.name}_final"
    output_dir.mkdir(parents=True, exist_ok=True)

    md_files = sorted(md_dir.glob("*.md"))

    print(f"Markdown: {md_dir} ({len(md_files)} files)")
    print(f"Tables:   {tables_dir}")
    print(f"Output:   {output_dir}")

    n_done = 0
    total_missing = 0

    for md_path in tqdm(md_files, desc="Recombine", unit="file"):
        result, n_missing = recombine_file(md_path, tables_dir)
        (output_dir / md_path.name).write_text(result, encoding="utf-8")
        total_missing += n_missing
        n_done += 1

    print(f"\nDone! {n_done} files recombined → {output_dir}")
    if total_missing:
        print(f"  {total_missing} tables had no OCR result (marked <!-- Missing table -->)")


if __name__ == "__main__":
    main()
