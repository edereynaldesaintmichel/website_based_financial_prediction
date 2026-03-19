#!/usr/bin/env python3
"""
Convert a directory of {stem}.html files into a single JSONL file.

Usage:
    python convert_tables_to_jsonl.py <tables_dir> [--output tables.jsonl]
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tables_dir", help="Directory of .html table files")
    parser.add_argument("--output", help="Output JSONL path (default: {tables_dir}.jsonl)")
    args = parser.parse_args()

    tables_dir = Path(args.tables_dir)
    if not tables_dir.is_dir():
        print(f"Error: {tables_dir} is not a directory")
        raise SystemExit(1)

    output_path = Path(args.output) if args.output else tables_dir.parent / f"{tables_dir.name}.jsonl"

    html_files = sorted(tables_dir.glob("*.html"))
    print(f"Found {len(html_files)} HTML files in {tables_dir}")
    print(f"Writing to {output_path}")

    with open(output_path, "w", encoding="utf-8") as out_f:
        for html_path in tqdm(html_files, desc="Convert", unit="file"):
            html = html_path.read_text(encoding="utf-8")
            out_f.write(json.dumps({"stem": html_path.stem, "html": html}, ensure_ascii=False) + "\n")

    print(f"Done → {output_path}")


if __name__ == "__main__":
    main()
