"""
Re-indent all single-line HTML files in a directory in-place.

Usage:
    python3 reindent_html.py [directory]

Default directory: /Users/eloireynal/Downloads/html_50plus_employees_sanitized
"""

import sys
from pathlib import Path
from bs4 import BeautifulSoup

DIRECTORY = "/Users/eloireynal/Downloads/html_50plus_employees_sanitized"


def reindent(path: Path) -> None:
    raw = path.read_text(encoding="utf-8", errors="replace")
    pretty = BeautifulSoup(raw, "html.parser").prettify()
    path.write_text(pretty, encoding="utf-8")


def main():
    directory = Path(sys.argv[1] if len(sys.argv) > 1 else DIRECTORY)
    if not directory.is_dir():
        print(f"Error: directory not found: {directory}")
        sys.exit(1)

    files = sorted(directory.glob("*.html"))
    print(f"Re-indenting {len(files)} files in {directory} ...\n")

    for i, f in enumerate(files, 1):
        try:
            reindent(f)
            print(f"[{i}/{len(files)}] ✓ {f.name}")
        except Exception as e:
            print(f"[{i}/{len(files)}] ✗ {f.name}: {e}")

    print(f"\n✅ Done!")


if __name__ == "__main__":
    main()
