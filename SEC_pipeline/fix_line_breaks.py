"""
Clean and fix formatting in SEC markdown files.

Fixes applied:
1. Join artificial mid-sentence line breaks (from fixed-width HTML columns)
2. Preserve markdown table rows (lines starting with |)
3. Remove header metadata (10-K / filename / Document preamble)
4. Remove orphan page numbers (standalone 1-999 on their own line)
5. Remove [Table of Contents] navigation links
6. Remove --- horizontal rules (page break artifacts)
7. Fix escaped underscores (\\_  ->  _)
"""

import argparse
import glob
import os
import re

# Matches a standalone page number (1-999), possibly surrounded by whitespace
RE_PAGE_NUMBER = re.compile(r"^\d{1,3}$")

# Matches [Table of Contents] links (with optional heading/bold prefix and anchor)
RE_TOC_LINK = re.compile(
    r"^[#*\s]*\[Table of Contents\]", re.IGNORECASE
)

# Matches --- horizontal rules (3+ dashes, optionally with spaces)
RE_HRULE = re.compile(r"^-{3,}\s*$")

# Matches a markdown table row: starts with |
RE_TABLE_ROW = re.compile(r"^\|")

# Header preamble lines to skip at the top of the file
# Common patterns: "10-K", "1", "filename.htm", "Document", "FORM 10-K", etc.
RE_HEADER_JUNK = re.compile(
    r"^("
    r"10-K(/A)?|"
    r"EX-\d+|"
    r"\d+|"                          # standalone number (page/sequence)
    r"\S+\.(htm|html|txt)|"          # filename
    r"(FORM\s+)?10-K(/A)?|"
    r"Document|"
    r"XBRL Viewer"
    r")$",
    re.IGNORECASE,
)


def strip_header(lines: list[str]) -> list[str]:
    """Remove the metadata preamble at the top of the file.

    Scans from the top and skips lines that match known header junk patterns
    or are blank, until we hit a line of real content.
    """
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "" or RE_HEADER_JUNK.match(stripped):
            start = i + 1
        else:
            break
    return lines[start:]


def clean_and_fix(text: str) -> str:
    lines = text.split("\n")

    # Phase 1: strip header preamble
    lines = strip_header(lines)

    # Phase 2: filter out noise lines and fix escapes
    cleaned = []
    for line in lines:
        stripped = line.strip()

        # Remove [Table of Contents] links
        if RE_TOC_LINK.match(stripped):
            continue

        # Remove --- horizontal rules
        if RE_HRULE.match(stripped):
            continue

        # Remove orphan page numbers (standalone 1-999)
        if RE_PAGE_NUMBER.match(stripped):
            continue

        # Fix escaped underscores
        line = line.replace("\\_", "_")
        cleaned.append(line)

    # Phase 3: join broken lines, but preserve table rows
    result = []
    buffer = []

    for line in cleaned:
        stripped = line.strip()

        if stripped == "":
            # Blank line = paragraph boundary
            if buffer:
                result.append(" ".join(buffer))
                buffer = []
            result.append("")
        elif RE_TABLE_ROW.match(stripped):
            # Table row — flush buffer first, then keep row as-is
            if buffer:
                result.append(" ".join(buffer))
                buffer = []
            result.append(stripped)
        else:
            buffer.append(stripped)

    # Flush remaining buffer
    if buffer:
        result.append(" ".join(buffer))

    # Phase 4: second-pass filter for TOC links that were split across lines
    # and only became visible after joining
    result = [
        line for line in result if not RE_TOC_LINK.match(line.strip())
    ]

    # Phase 5: collapse runs of 3+ blank lines into 2
    final = []
    blank_count = 0
    for line in result:
        if line == "":
            blank_count += 1
            if blank_count <= 2:
                final.append(line)
        else:
            blank_count = 0
            final.append(line)

    # Strip leading/trailing blank lines
    text_out = "\n".join(final).strip()
    return text_out + "\n" if text_out else ""


def main():
    parser = argparse.ArgumentParser(
        description="Clean and fix formatting in SEC markdown files"
    )
    parser.add_argument("input_dir", help="Directory containing markdown files")
    parser.add_argument(
        "--output-dir",
        help="Write fixed files to this directory (default: overwrite in place)",
    )
    args = parser.parse_args()

    md_files = glob.glob(os.path.join(args.input_dir, "*.md"))
    print(f"Found {len(md_files)} markdown files")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    modified = 0
    for filepath in sorted(md_files):
        with open(filepath, "r", encoding="utf-8") as f:
            original = f.read()

        fixed = clean_and_fix(original)

        if fixed != original:
            modified += 1

        out_path = (
            os.path.join(args.output_dir, os.path.basename(filepath))
            if args.output_dir
            else filepath
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(fixed)

    print(
        f"Wrote {len(md_files)} files ({modified} modified) "
        f"to {args.output_dir or args.input_dir}"
    )


if __name__ == "__main__":
    main()
