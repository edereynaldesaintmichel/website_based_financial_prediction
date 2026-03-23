"""
Step 3: Bucket tokenized chunks by sequence length.

Computes optimal bucket boundaries from the data to minimize
intra-bucket length variance (1D k-means / Fisher-Jenks style),
then pads and stacks each bucket into a single .pt file with
pre-packed tensors ready for training.

Output per bucket:
    bucket_{bound}.pt = {
        "input_ids":      (N, pad_to)    uint16
        "is_number_mask": (N, pad_to)    int8
        "number_values":  (N, pad_to)    float32
        "source_files":   [str, ...]
        "pad_to":         int
    }

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

import torch
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


def pad_and_stack(items: list, pad_to: int) -> dict:
    """Pad variable-length items and stack into contiguous tensors."""
    n = len(items)
    input_ids = torch.zeros(n, pad_to, dtype=torch.uint16)
    is_number_mask = torch.zeros(n, pad_to, dtype=torch.int8)
    number_values = torch.zeros(n, pad_to, dtype=torch.float32)
    table_row_index = torch.zeros(n, pad_to, dtype=torch.int16)
    table_col_index = torch.zeros(n, pad_to, dtype=torch.int16)
    table_mask = torch.zeros(n, pad_to, dtype=torch.int8)
    table_num_rows = torch.zeros(n, pad_to, dtype=torch.int16)
    table_num_cols = torch.zeros(n, pad_to, dtype=torch.int16)
    source_files = []

    for i, item in enumerate(items):
        ids = item["input_ids"]
        seq_len = len(ids)
        input_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.uint16)
        is_number_mask[i, :seq_len] = torch.tensor(item["is_number_mask"], dtype=torch.int8)
        number_values[i, :seq_len] = torch.tensor(item["number_values"][:seq_len], dtype=torch.float32)
        if "table_row_index" in item:
            table_row_index[i, :seq_len] = torch.tensor(item["table_row_index"][:seq_len], dtype=torch.int16)
            table_col_index[i, :seq_len] = torch.tensor(item["table_col_index"][:seq_len], dtype=torch.int16)
            table_mask[i, :seq_len] = torch.tensor(item["table_mask"][:seq_len], dtype=torch.int8)
            table_num_rows[i, :seq_len] = torch.tensor(item["table_num_rows"][:seq_len], dtype=torch.int16)
            table_num_cols[i, :seq_len] = torch.tensor(item["table_num_cols"][:seq_len], dtype=torch.int16)
        source_files.append(item.get("source_file", ""))

    return {
        "input_ids": input_ids,
        "is_number_mask": is_number_mask,
        "number_values": number_values,
        "table_row_index": table_row_index,
        "table_col_index": table_col_index,
        "table_mask": table_mask,
        "table_num_rows": table_num_rows,
        "table_num_cols": table_num_cols,
        "source_files": source_files,
        "pad_to": pad_to,
    }


def main():
    parser = argparse.ArgumentParser(description="Bucket tokenized chunks by length")
    parser.add_argument("--input", required=True, help="Input tokenized JSONL file")
    parser.add_argument("--output_dir", required=True, help="Output directory for bucketed .pt files")
    parser.add_argument("--num_buckets", type=int, default=10,
                        help="Number of buckets (boundaries computed to minimize variance)")
    parser.add_argument("--buckets", nargs="+", type=int, default=None,
                        help="Manual bucket upper bounds (overrides --num_buckets)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Pass 1: collect lengths
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

    # Pass 2: spool items to temporary per-bucket JSONL files on disk
    # (avoids holding all items in memory at once)
    import shutil
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="bucket_tmp_")

    try:
        bucket_tmp_files = {}
        bucket_counts = defaultdict(int)
        bucket_num_counts = defaultdict(int)
        bucket_lengths = defaultdict(list)

        for bound in bucket_bounds:
            tmp_path = os.path.join(tmp_dir, f"tmp_{bound}.jsonl")
            bucket_tmp_files[bound] = open(tmp_path, "w", encoding="utf-8")

        with open(args.input, "r", encoding="utf-8") as f:
            for line in tqdm(f, total=len(lengths), desc="Pass 2: bucketing to disk", unit="seq"):
                item = json.loads(line)
                seq_len = item["seq_length"]
                bound = get_bucket(seq_len, bucket_bounds)
                bucket_tmp_files[bound].write(line)
                bucket_counts[bound] += 1
                bucket_num_counts[bound] += sum(item["is_number_mask"])
                bucket_lengths[bound].append(seq_len)

        for fh in bucket_tmp_files.values():
            fh.close()

        # Pass 3: load each bucket independently, pad+stack, save .pt, free memory
        total_variance = 0
        total_seqs = 0
        non_empty = 0
        for bound in bucket_bounds:
            count = bucket_counts[bound]
            if count == 0:
                continue

            non_empty += 1
            total_seqs += count
            bl = bucket_lengths[bound]
            mean_len = sum(bl) / count
            variance = sum((l - mean_len) ** 2 for l in bl) / count
            padding_waste = sum(bound - l for l in bl) / (bound * count) * 100
            total_variance += variance * count

            print(f"  Bucket ≤{bound:>4}: {count:>6} seqs | "
                  f"len {min(bl):>3}-{max(bl):>3} | "
                  f"mean {mean_len:>5.0f} | var {variance:>8.1f} | "
                  f"pad waste {padding_waste:>4.1f}% | "
                  f"numbers {bucket_num_counts[bound]:>6}")

            # Load this bucket's items from temp file
            tmp_path = os.path.join(tmp_dir, f"tmp_{bound}.jsonl")
            items = []
            with open(tmp_path, "r", encoding="utf-8") as f:
                for line in f:
                    items.append(json.loads(line))

            # Stack into tensors and save
            bucket_data = pad_and_stack(items, bound)
            del items  # free raw dicts before saving
            output_path = os.path.join(args.output_dir, f"bucket_{bound}.pt")
            torch.save(bucket_data, output_path)
            del bucket_data  # free tensors before next bucket
            mb = os.path.getsize(output_path) / 1e6
            print(f"           -> {output_path} ({mb:.1f} MB)")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    avg_variance = total_variance / total_seqs
    print(f"\nTotal: {total_seqs} sequences, {non_empty} non-empty buckets")
    print(f"Weighted avg intra-bucket variance: {avg_variance:.1f}")
    print(f"Saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
