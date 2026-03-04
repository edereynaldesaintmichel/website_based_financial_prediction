"""
Shared helpers for length-corrected information-density scoring.

The metric is:
    adjusted_density = log(raw_zlib_ratio / baseline_ratio(byte_length))

where baseline_ratio(n) is log-linearly interpolated from a reference curve
built from natural-language Wikipedia text (see build_baseline.py).

This removes the systematic compression advantage that longer pages have:
zlib finds more back-references in longer text, so a long-but-repetitive page
would naively score similar to a short-but-dense page.  The corrected score
is always relative to what typical text of the same length would compress to.

Using a log ratio (rather than a plain difference) keeps the score symmetric:
doubling and halving the expected ratio produce equal-magnitude offsets, and
the natural zero point is 0.0 (exactly average).
"""
import json
import math
import zlib
from pathlib import Path

MIN_CONTENT_BYTES = 50  # pages shorter than this are pinned to -1.0


def raw_ratio(content: str):
    """Return (ratio, n_bytes); ratio is None if content is too short."""
    raw = content.encode("utf-8")
    n = len(raw)
    if n < MIN_CONTENT_BYTES:
        return None, n
    return len(zlib.compress(raw, 6)) / n, n


def load_baseline(path) -> list:
    """Load baseline curve from JSON. Returns [(n_bytes, ratio), ...] sorted."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [(int(entry[0]), float(entry[1])) for entry in data["curve"]]


def interp_expected_ratio(n_bytes: int, curve: list) -> float:
    """
    Log-linear interpolation of the expected zlib ratio at n_bytes.
    Clamps to the curve's endpoints outside the measured range.
    """
    if not curve:
        return 0.5
    if n_bytes <= curve[0][0]:
        return curve[0][1]
    if n_bytes >= curve[-1][0]:
        return curve[-1][1]
    lo, hi = 0, len(curve) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if curve[mid][0] <= n_bytes:
            lo = mid
        else:
            hi = mid
    n0, r0 = curve[lo]
    n1, r1 = curve[hi]
    t = (math.log(n_bytes) - math.log(n0)) / (math.log(n1) - math.log(n0))
    return r0 + t * (r1 - r0)


def adjusted_density(content: str, curve: list) -> float:
    """
    Length-corrected information density.

    Returns log(raw_zlib_ratio / baseline_ratio(byte_length)).
      0.0       → exactly average density for this length.
      Positive  → more information-dense than typical text of that length.
      Negative  → more compressible / redundant than typical text of that length.
      -1.0      → near-empty page (< MIN_CONTENT_BYTES).
    """
    ratio, n = raw_ratio(content)
    if ratio is None:
        return -1.0
    expected = interp_expected_ratio(n, curve)
    if expected <= 0:
        return -1.0
    return math.log(ratio / expected)
