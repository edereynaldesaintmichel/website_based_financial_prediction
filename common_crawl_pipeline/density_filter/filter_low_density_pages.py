"""
Remove pages with an adjusted density score below a fixed threshold from
cc_data/markdown/ .md files.

Density = log(raw_zlib_ratio / baseline_ratio(byte_length)), where the
baseline is the average compression ratio of natural-language Wikipedia text
at that byte length (built by build_baseline.py).  This removes the
systematic advantage longer pages have under raw compression ratio.

Pages with adjusted density < --threshold are discarded (default: -0.4).

Usage:
  # Build the baseline once:
  python build_baseline.py

  python filter_low_density_pages.py [--threshold -0.4]
                                     [--markdown-dir cc_data/markdown]
                                     [--baseline baseline_curve.json]
                                     [--dry-run]
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

# Allow running from any directory by adding this script's folder to sys.path.
sys.path.insert(0, str(Path(__file__).parent))
from density_utils import adjusted_density, load_baseline  # noqa: E402

SEP_RE = re.compile(r'(?m)^(########## .+)$')

DEFAULT_BASELINE = Path(__file__).parent / "baseline_curve.json"
DEFAULT_THRESHOLD = -0.4


def parse_pages(text: str) -> list[tuple[str, str]]:
    """Split a .md file into (separator_line, content) pairs."""
    matches = list(SEP_RE.finditer(text))
    pages = []
    for i, m in enumerate(matches):
        sep = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        pages.append((sep, text[start:end]))
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Discard pages with adjusted density below this value (default: -0.4)")
    parser.add_argument("--markdown-dir", default="cc_data/markdown",
                        help="Directory containing .md files (default: cc_data/markdown)")
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE),
                        help="Path to baseline_curve.json (default: alongside this script)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without modifying any files")
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline curve not found: {baseline_path}\n"
            f"Run build_baseline.py first."
        )
    curve = load_baseline(baseline_path)

    md_dir = Path(args.markdown_dir)
    threshold = args.threshold

    # ── Score all pages globally ───────────────────────────────────────────────
    print("Scoring pages…")
    all_entries: list[tuple[float, Path, str, str]] = []
    for path in sorted(md_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for sep, content in parse_pages(text):
            all_entries.append((adjusted_density(content, curve), path, sep, content))

    if not all_entries:
        print("No pages found.")
        return

    total = len(all_entries)
    to_remove = sum(1 for d, *_ in all_entries if d < threshold)

    print(f"Total pages     : {total}")
    print(f"Threshold       : adjusted density < {threshold}")
    print(f"Pages to remove : {to_remove}")
    print(f"Pages to keep   : {total - to_remove}")

    if args.dry_run:
        print("Dry-run — no files modified.")
        return

    # ── Rebuild files ─────────────────────────────────────────────────────────
    file_pages: dict[Path, list[tuple[float, str, str]]] = defaultdict(list)
    for d, path, sep, content in all_entries:
        file_pages[path].append((d, sep, content))

    deleted_files = 0
    for path, entries in file_pages.items():
        kept = [(sep, content) for d, sep, content in entries if d >= threshold]
        if not kept:
            path.unlink()
            deleted_files += 1
        else:
            path.write_text("".join(sep + content for sep, content in kept),
                            encoding="utf-8")

    print(f"Files deleted   : {deleted_files}")
    print("Done.")


if __name__ == "__main__":
    main()
