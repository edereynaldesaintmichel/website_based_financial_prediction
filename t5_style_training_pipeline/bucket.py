"""
Unified bucketing: merge all tokenized_{size}.jsonl into shared buckets.

Sequences from all chunk sizes are pooled together, then bucketed by
actual tokenized length. This means bucket_128.pt may contain sequences
from chunks_128, chunks_256, etc. that happened to tokenize to ≤128 tokens.

Usage:
    python -m t5_style_training_pipeline.bucket \
        --input_dir t5_training_data/tokenized \
        --output_dir t5_training_data/bucketed \
        --num_buckets 10
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlm_training_pipeline.bucket_by_length import (
    compute_optimal_boundaries,
    get_bucket,
    pad_and_stack,
)


def main():
    parser = argparse.ArgumentParser(description="Unified bucketing across chunk sizes")
    parser.add_argument("--input_dir", default="t5_training_data/tokenized")
    parser.add_argument("--output_dir", default="t5_training_data/bucketed")
    parser.add_argument("--num_buckets", type=int, default=10)
    parser.add_argument("--buckets", nargs="+", type=int, default=None,
                        help="Manual bucket upper bounds (overrides --num_buckets)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenized_files = sorted(Path(args.input_dir).glob("tokenized_*.jsonl"))
    if not tokenized_files:
        print(f"No tokenized_*.jsonl files found in {args.input_dir}")
        return

    # Pass 1: collect lengths from all files
    lengths = []
    file_counts = {}
    for tf in tokenized_files:
        count = 0
        with open(tf, "r", encoding="utf-8") as f:
            for line in f:
                lengths.append(json.loads(line)["seq_length"])
                count += 1
        file_counts[tf.name] = count
        print(f"  {tf.name}: {count} sequences")

    print(f"\nTotal: {len(lengths)} sequences (lengths {min(lengths)}-{max(lengths)})")

    if args.buckets:
        bucket_bounds = sorted(args.buckets)
        print(f"Using manual buckets: {bucket_bounds}")
    else:
        bucket_bounds = compute_optimal_boundaries(lengths, args.num_buckets)
        print(f"Optimal buckets ({len(bucket_bounds)}): {bucket_bounds}")

    # Pass 2: spool to temp files
    tmp_dir = tempfile.mkdtemp(prefix="t5_bucket_tmp_")
    try:
        bucket_tmp_files = {}
        bucket_counts = defaultdict(int)
        bucket_num_counts = defaultdict(int)
        bucket_lengths = defaultdict(list)

        for bound in bucket_bounds:
            bucket_tmp_files[bound] = open(
                os.path.join(tmp_dir, f"tmp_{bound}.jsonl"), "w", encoding="utf-8"
            )

        for tf in tokenized_files:
            with open(tf, "r", encoding="utf-8") as f:
                for line in tqdm(f, desc=f"Bucketing {tf.name}"):
                    item = json.loads(line)
                    seq_len = item["seq_length"]
                    bound = get_bucket(seq_len, bucket_bounds)
                    bucket_tmp_files[bound].write(line)
                    bucket_counts[bound] += 1
                    bucket_num_counts[bound] += sum(item["is_number_mask"])
                    bucket_lengths[bound].append(seq_len)

        for fh in bucket_tmp_files.values():
            fh.close()

        # Pass 3: pad, stack, save
        total_seqs = 0
        for bound in bucket_bounds:
            count = bucket_counts[bound]
            if count == 0:
                continue
            total_seqs += count
            bl = bucket_lengths[bound]
            mean_len = sum(bl) / count
            padding_waste = sum(bound - l for l in bl) / (bound * count) * 100

            print(f"  Bucket ≤{bound:>4}: {count:>6} seqs | "
                  f"len {min(bl):>3}-{max(bl):>3} | mean {mean_len:>5.0f} | "
                  f"pad waste {padding_waste:>4.1f}% | numbers {bucket_num_counts[bound]:>6}")

            items = []
            with open(os.path.join(tmp_dir, f"tmp_{bound}.jsonl"), "r", encoding="utf-8") as f:
                for line in f:
                    items.append(json.loads(line))

            bucket_data = pad_and_stack(items, bound)
            del items
            output_path = os.path.join(args.output_dir, f"bucket_{bound}.pt")
            torch.save(bucket_data, output_path)
            del bucket_data
            mb = os.path.getsize(output_path) / 1e6
            print(f"           -> {output_path} ({mb:.1f} MB)")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\nDone. {total_seqs} sequences saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
