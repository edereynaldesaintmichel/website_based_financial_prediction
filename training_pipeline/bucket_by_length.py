"""
Step 3: Bucket tokenized chunks by sequence length.

Computes optimal bucket boundaries from the data to minimize
intra-bucket length variance (1D k-means / Fisher-Jenks style),
then saves each bucket as a separate file.

Usage:
    python -m training_pipeline.bucket_by_length \
        --input training_data/tokenized/tokenized.jsonl \
        --output_dir training_data/bucketed \
        --num_buckets 10
"""
import argparse
import json
import os
from collections import defaultdict

from tqdm import tqdm


def compute_optimal_boundaries(lengths: list, num_buckets: int) -> list:
    """
    Find bucket upper-bounds that minimize total intra-bucket variance.

    Uses dynamic programming (Fisher / Jenks natural breaks) on sorted lengths.
    O(n² · k) but n = number of distinct lengths, which is at most ~512.
    """
    sorted_vals = sorted(set(lengths))
    n = len(sorted_vals)

    if n <= num_buckets:
        return sorted_vals

    # Frequency map
    freq = defaultdict(int)
    for l in lengths:
        freq[l] += 1

    # prefix sums for weighted mean / variance
    w = [freq[v] for v in sorted_vals]       # weights (counts)
    s = [freq[v] * v for v in sorted_vals]   # weighted sums
    s2 = [freq[v] * v * v for v in sorted_vals]  # weighted sum of squares

    # cumulative
    cum_w = [0] * (n + 1)
    cum_s = [0] * (n + 1)
    cum_s2 = [0] * (n + 1)
    for i in range(n):
        cum_w[i + 1] = cum_w[i] + w[i]
        cum_s[i + 1] = cum_s[i] + s[i]
        cum_s2[i + 1] = cum_s2[i] + s2[i]

    def range_variance(lo, hi):
        """Total weighted SSE for sorted_vals[lo..hi] inclusive."""
        cw = cum_w[hi + 1] - cum_w[lo]
        if cw == 0:
            return 0.0
        cs = cum_s[hi + 1] - cum_s[lo]
        cs2 = cum_s2[hi + 1] - cum_s2[lo]
        mean = cs / cw
        return cs2 - 2 * mean * cs + mean * mean * cw

    # DP: dp[k][i] = min total SSE partitioning sorted_vals[0..i] into k groups
    INF = float("inf")
    dp = [[INF] * n for _ in range(num_buckets + 1)]
    split = [[0] * n for _ in range(num_buckets + 1)]

    for i in range(n):
        dp[1][i] = range_variance(0, i)

    for k in range(2, num_buckets + 1):
        for i in range(k - 1, n):
            for j in range(k - 2, i):
                cost = dp[k - 1][j] + range_variance(j + 1, i)
                if cost < dp[k][i]:
                    dp[k][i] = cost
                    split[k][i] = j

    # Trace back boundaries
    boundaries = []
    k = num_buckets
    i = n - 1
    while k > 1:
        j = split[k][i]
        boundaries.append(sorted_vals[i])
        i = j
        k -= 1
    boundaries.append(sorted_vals[i])
    boundaries.reverse()

    return boundaries


def get_bucket(seq_length: int, bucket_bounds: list) -> int:
    """Return the bucket upper bound for a given sequence length."""
    for bound in bucket_bounds:
        if seq_length <= bound:
            return bound
    return bucket_bounds[-1]


def main():
    parser = argparse.ArgumentParser(description="Bucket tokenized chunks by length")
    parser.add_argument("--input", required=True, help="Input tokenized JSONL file")
    parser.add_argument("--output_dir", required=True, help="Output directory for bucketed files")
    parser.add_argument("--num_buckets", type=int, default=10,
                        help="Number of buckets (boundaries computed to minimize variance)")
    parser.add_argument("--buckets", nargs="+", type=int, default=None,
                        help="Manual bucket upper bounds (overrides --num_buckets)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Pass 1: stream file to collect only seq_length values (low RAM)
    lengths = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Pass 1: reading lengths", unit="seq"):
            lengths.append(json.loads(line)["seq_length"])

    print(f"Found {len(lengths)} sequences (lengths {min(lengths)}-{max(lengths)})")

    # Compute or use bucket boundaries
    if args.buckets:
        bucket_bounds = sorted(args.buckets)
        print(f"Using manual buckets: {bucket_bounds}")
    else:
        bucket_bounds = compute_optimal_boundaries(lengths, args.num_buckets)
        print(f"Optimal buckets ({len(bucket_bounds)}): {bucket_bounds}")

    # Pass 2: stream file again, write each item directly to its bucket file
    bucket_files = {}
    bucket_lengths = defaultdict(list)
    bucket_num_counts = defaultdict(int)

    for bound in bucket_bounds:
        output_file = os.path.join(args.output_dir, f"bucket_{bound}.jsonl")
        bucket_files[bound] = open(output_file, "w", encoding="utf-8")

    with open(args.input, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=len(lengths), desc="Pass 2: bucketing", unit="seq"):
            item = json.loads(line)
            seq_len = item["seq_length"]
            bound = get_bucket(seq_len, bucket_bounds)
            bucket_files[bound].write(line)
            bucket_lengths[bound].append(seq_len)
            bucket_num_counts[bound] += sum(item["is_number_mask"])

    for fh in bucket_files.values():
        fh.close()

    # Report stats
    total_variance = 0
    total_seqs = 0
    non_empty = 0
    for bound in bucket_bounds:
        bl = bucket_lengths[bound]
        if not bl:
            continue
        non_empty += 1
        total_seqs += len(bl)
        mean_len = sum(bl) / len(bl)
        variance = sum((l - mean_len) ** 2 for l in bl) / len(bl)
        padding_waste = sum(bound - l for l in bl) / (bound * len(bl)) * 100
        total_variance += variance * len(bl)

        print(f"  Bucket ≤{bound:>4}: {len(bl):>6} seqs | "
              f"len {min(bl):>3}-{max(bl):>3} | "
              f"mean {mean_len:>5.0f} | var {variance:>8.1f} | "
              f"pad waste {padding_waste:>4.1f}% | "
              f"numbers {bucket_num_counts[bound]:>6}")

    avg_variance = total_variance / total_seqs
    print(f"\nTotal: {total_seqs} sequences, {non_empty} non-empty buckets")
    print(f"Weighted avg intra-bucket variance: {avg_variance:.1f}")
    print(f"Saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
