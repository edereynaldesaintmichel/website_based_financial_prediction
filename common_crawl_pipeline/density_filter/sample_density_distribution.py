"""
Sample pages from non-overlapping density bands for visual inspection.

Density = raw_zlib_ratio − baseline_ratio(byte_length), where the baseline
is the average compression ratio of natural-language Wikipedia text at that
byte length (built by build_baseline.py).  This removes the systematic
advantage longer pages have under raw compression ratio.

  Positive density → denser than typical natural text of the same length.
  Negative density → more compressible/redundant than expected.

Bands (non-overlapping, cover 0–100%):
  Bottom 5%  :  [0%,  5%)
  5–10%      :  [5%, 10%)
  10–20%     : [10%, 20%)
  20–40%     : [20%, 40%)
  40–80%     : [40%, 80%)
  Top 20%    : [80%,100%]

Usage:
  # Build the baseline once:
  python build_baseline.py

  python sample_density_distribution.py [--markdown-dir cc_data/markdown]
                                        [--output density_samples.md]
                                        [--samples-per-band 5]
                                        [--max-chars 12000]
                                        [--baseline baseline_curve.json]
"""
import argparse
import re
import sys
from pathlib import Path

# Allow running from any directory by adding this script's folder to sys.path.
sys.path.insert(0, str(Path(__file__).parent))
from density_utils import adjusted_density, load_baseline  # noqa: E402

SEP_RE = re.compile(r'(?m)^(########## .+)$')

BANDS = [
    ("Bottom 5%",  0.00, 0.05),
    ("5–10%",      0.05, 0.10),
    ("10–20%",     0.10, 0.20),
    ("20–40%",     0.20, 0.40),
    ("40–80%",     0.40, 0.80),
    ("Top 20%",    0.80, 1.00),
]

DEFAULT_BASELINE = Path(__file__).parent / "baseline_curve.json"


def parse_pages(text: str):
    matches = list(SEP_RE.finditer(text))
    for i, m in enumerate(matches):
        sep = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        yield sep, text[start:end]


def evenly_spaced(lst: list, k: int) -> list:
    """Pick k evenly-spaced elements from lst (no randomness — reproducible)."""
    if k >= len(lst):
        return lst
    return [lst[int(len(lst) * (i + 0.5) / k)] for i in range(k)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--markdown-dir", default="cc_data/markdown")
    parser.add_argument("--output", default="density_samples.md")
    parser.add_argument("--samples-per-band", type=int, default=5)
    parser.add_argument("--max-chars", type=int, default=12000,
                        help="Truncate page content to this many chars in the output")
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE),
                        help="Path to baseline_curve.json (default: alongside this script)")
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline curve not found: {baseline_path}\n"
            f"Run build_baseline.py first."
        )
    curve = load_baseline(baseline_path)

    md_dir = Path(args.markdown_dir)

    # ── Score all pages ────────────────────────────────────────────────────────
    print("Scoring pages…")
    all_pages: list[tuple[float, str, str]] = []  # (density, sep, content)
    for path in sorted(md_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for sep, content in parse_pages(text):
            all_pages.append((adjusted_density(content, curve), sep, content))

    all_pages.sort(key=lambda x: x[0])
    total = len(all_pages)
    total_chars = sum(len(sep) + len(content) for _, sep, content in all_pages)
    print(f"Total pages: {total}")

    # ── Build output ───────────────────────────────────────────────────────────
    lines = [
        "# Density Distribution Samples\n\n",
        f"**Total pages scored:** {total}  \n",
        f"**Total characters (all bands):** {total_chars:,}  \n",
        "**Metric:** `raw_zlib_ratio − baseline_ratio(byte_length)`  \n",
        "(positive = denser than natural text of the same length; "
        "negative = more compressible/redundant)  \n",
        f"**Baseline:** `{baseline_path.name}` ({len(curve)} reference points)\n\n",
        "---\n",
    ]

    for band_name, lo, hi in BANDS:
        lo_idx = int(total * lo)
        hi_idx = int(total * hi)          # exclusive upper bound
        band = all_pages[lo_idx:hi_idx]

        lo_d = band[0][0]  if band else float("nan")
        hi_d = band[-1][0] if band else float("nan")

        band_chars = sum(len(sep) + len(content) for _, sep, content in band)
        char_share = band_chars / total_chars * 100 if total_chars else 0.0

        lines.append(
            f"\n## {band_name}  "
            f"(density {lo_d:.4f} – {hi_d:.4f} | {len(band)} pages | "
            f"{band_chars:,} chars, {char_share:.1f}% of total)\n\n"
        )

        if not band:
            lines.append("_No pages in this band._\n")
            continue

        samples = evenly_spaced(band, args.samples_per_band)

        for i, (d, sep, content) in enumerate(samples, 1):
            snippet = content.strip()
            truncated = False
            if len(snippet) > args.max_chars:
                snippet = snippet[:args.max_chars]
                truncated = True

            lines.append(f"### Sample {i}  —  density: **{d:.4f}**\n\n")
            lines.append(f"`{sep}`\n\n")
            lines.append("```\n")
            lines.append(snippet + "\n")
            if truncated:
                lines.append(f"… [truncated — full page: {len(content.strip())} chars]\n")
            lines.append("```\n\n")

    out_path = Path(args.output)
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
