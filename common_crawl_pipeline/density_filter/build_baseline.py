"""
Build a reference compression-ratio vs. byte-length baseline curve.

Fetches Wikipedia articles and concatenates them into --n-samples large blobs,
each targeting --target-bytes (~1 MB by default).  Articles are drawn from a
broad pool covering history, science, culture, and geography so each blob is
independently diverse natural prose.

For each blob and for each of --n-points log-spaced byte lengths in
[100, --target-bytes], we measure the zlib compression ratio of the blob prefix
at that length, then average across all blobs to get the expected ratio at each
length.

The result is a JSON file used by sample_density_distribution.py and
filter_low_density_pages.py for log-linear interpolation.

Usage:
  python build_baseline.py [--output baseline_curve.json]
                           [--target-bytes 1000000]
                           [--n-samples 10]
                           [--n-points 100]
"""
import argparse
import json
import math
import urllib.parse
import urllib.request
import zlib
from collections import defaultdict
from pathlib import Path

# Large pool of diverse natural-prose Wikipedia articles.
# ~100 titles so 10 × 1 MB blobs can each draw from non-overlapping subsets.
WIKI_ARTICLE_POOL = [
    # History & Politics
    "History_of_the_United_States", "World_War_II", "Roman_Empire",
    "French_Revolution", "Cold_War", "Ancient_Greece", "Byzantine_Empire",
    "British_Empire", "American_Civil_War", "Russian_Revolution",
    "Mongol_Empire", "Ottoman_Empire", "History_of_China", "Ancient_Egypt",
    "History_of_India", "Colonialism", "Age_of_Enlightenment",
    "World_War_I", "History_of_Europe", "History_of_Africa",
    # Science & Technology
    "Science", "Quantum_mechanics", "Theory_of_relativity", "Evolution",
    "DNA", "Neuroscience", "Artificial_intelligence", "Internet",
    "History_of_computing", "Space_exploration", "Astronomy", "Biology",
    "Chemistry", "Physics", "Mathematics", "Ecology", "Geology",
    "Oceanography", "Meteorology", "Thermodynamics",
    # Medicine & Life sciences
    "Medicine", "Human_body", "Genetics", "Immunology", "Epidemiology",
    "Neurology", "Psychology", "Psychiatry", "Pharmacology", "Cancer",
    # Culture & Society
    "Philosophy", "Economics", "Sociology", "Religion", "Art",
    "Music", "Literature", "Language", "Linguistics", "Anthropology",
    "Archaeology", "Architecture", "Cinema", "Theatre", "Photography",
    "Ethics", "Political_science", "Law", "Education", "Journalism",
    # Geography & Environment
    "Climate_change", "Geography", "Atmosphere_of_Earth", "Ocean",
    "Amazon_rainforest", "Sahara", "Arctic", "Antarctica",
    "Himalaya", "Mediterranean_Sea",
    # More history & civilisation
    "Renaissance", "Industrial_Revolution", "Middle_Ages",
    "Ancient_Rome", "Mesopotamia", "Maya_civilization",
    "Inca_Empire", "Aztec_Empire", "Viking_Age", "Crusades",
    "History_of_Japan", "History_of_Korea", "History_of_Russia",
    "History_of_Brazil", "History_of_Australia",
    # Economics & Society
    "Capitalism", "Democracy", "Globalization", "Urbanization",
    "Human_rights", "Feminism", "Nationalism",
    # Natural sciences cont.
    "Human_evolution", "Plate_tectonics", "Natural_selection",
    "Big_Bang", "Black_hole", "Particle_physics", "Electromagnetism",
    "Fluid_dynamics", "Quantum_field_theory", "General_relativity",
    # Technology & Engineering
    "Telecommunications", "Semiconductor", "Renewable_energy",
    "Nuclear_power", "Robotics", "Biotechnology", "Nanotechnology",
]

MIN_BYTES = 100


def fetch_wikipedia_text(title: str) -> str:
    """Fetch the plain-text extract of a Wikipedia article via the API."""
    params = urllib.parse.urlencode({
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": "1",
        "format": "json",
        "redirects": "1",
    })
    url = f"https://en.wikipedia.org/w/api.php?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "density-baseline-builder/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read())
    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    return page.get("extract", "")


def log_spaced_lengths(n_points: int, min_b: int, max_b: int) -> list:
    """Return n_points unique integers log-spaced in [min_b, max_b]."""
    log_min = math.log(min_b)
    log_max = math.log(max_b)
    raw = [
        int(round(math.exp(log_min + (log_max - log_min) * i / (n_points - 1))))
        for i in range(n_points)
    ]
    return sorted(set(raw))


def build_blobs(n_samples: int, target_bytes: int) -> list[bytes]:
    """
    Fetch articles from WIKI_ARTICLE_POOL and concatenate them into n_samples
    blobs, each at least target_bytes long.  Articles are assigned round-robin
    so every blob gets a different mix of topics.
    """
    # Pre-fetch all articles once, in order.
    print(f"Fetching up to {len(WIKI_ARTICLE_POOL)} Wikipedia articles…")
    article_bytes: list[bytes] = []
    for title in WIKI_ARTICLE_POOL:
        print(f"  {title}… ", end="", flush=True)
        try:
            text = fetch_wikipedia_text(title)
            b = text.encode("utf-8")
            print(f"{len(b):,} bytes")
            article_bytes.append(b)
        except Exception as e:
            print(f"FAILED ({e})")

    if not article_bytes:
        return []

    # Distribute articles round-robin across blobs, skipping blobs that have
    # already reached target_bytes so we stop fetching as soon as all are full.
    parts: list[list[bytes]] = [[] for _ in range(n_samples)]
    sizes: list[int] = [0] * n_samples
    blob_idx = 0
    for ab in article_bytes:
        # Advance to the next blob that still needs more content.
        for _ in range(n_samples):
            if sizes[blob_idx] < target_bytes:
                break
            blob_idx = (blob_idx + 1) % n_samples
        else:
            break  # All blobs have reached target_bytes.
        parts[blob_idx].append(ab)
        sizes[blob_idx] += len(ab)
        blob_idx = (blob_idx + 1) % n_samples

    blobs = []
    for i, part_list in enumerate(parts):
        blob = b"\n\n".join(part_list)
        print(f"  Blob {i + 1}: {len(blob):,} bytes from {len(part_list)} articles")
        if len(blob) < MIN_BYTES:
            print(f"    (skipped — too short)")
            continue
        blobs.append(blob)

    return blobs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", default="baseline_curve.json",
                        help="Output JSON path (default: baseline_curve.json)")
    parser.add_argument("--target-bytes", type=int, default=1_000_000,
                        help="Target size of each blob in bytes (default: 1000000)")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Number of independent blobs to build (default: 10)")
    parser.add_argument("--n-points", type=int, default=100,
                        help="Number of log-spaced measurement points (default: 100)")
    args = parser.parse_args()

    # ── Build reference blobs ─────────────────────────────────────────────────
    print(f"\nBuilding {args.n_samples} blobs targeting {args.target_bytes:,} bytes each…\n")
    samples = build_blobs(args.n_samples, args.target_bytes)

    if not samples:
        print("No samples retrieved — aborting.")
        return

    actual_max = min(args.target_bytes, max(len(s) for s in samples))
    lengths = log_spaced_lengths(args.n_points, MIN_BYTES, actual_max)

    print(
        f"\nMeasuring at {len(lengths)} log-spaced lengths "
        f"({MIN_BYTES} – {actual_max:,} bytes) across {len(samples)} blobs…"
    )

    # ── Average compression ratios ────────────────────────────────────────────
    sum_ratios: dict = defaultdict(float)
    count_ratios: dict = defaultdict(int)

    for sample in samples:
        for n in lengths:
            if n > len(sample):
                break
            ratio = len(zlib.compress(sample[:n], 6)) / n
            sum_ratios[n] += ratio
            count_ratios[n] += 1

    curve = sorted(
        [[n, sum_ratios[n] / count_ratios[n]] for n in sum_ratios],
        key=lambda x: x[0],
    )

    # Report how many samples contributed to the longest points
    last_n, _ = curve[-1]
    last_count = count_ratios[last_n]
    print(f"Curve has {len(curve)} points "
          f"(longest point uses {last_count}/{len(samples)} blobs).")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.write_text(json.dumps({"curve": curve}, indent=2), encoding="utf-8")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
