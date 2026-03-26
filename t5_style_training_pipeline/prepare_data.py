"""
Prepare data for T5-style CLS training: chunk → tokenize → holdout split → bucket.

Multi-scale chunking (128/256/512/1024 tokens) so the CLS token learns to
compress varying sequence lengths. 10% of source files held out for validation.

Uses functions from the MLM pipeline — no duplication.

Usage:
    python -m t5_style_training_pipeline.prepare_data \
        --input_dirs training_data/processed/SEC_10k_markdown_tagged \
                     training_data/processed/companies_house_markdown_tagged \
        --output_dir t5_training_data
"""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlm_training_pipeline.chunk_markdown import process_directory
from mlm_training_pipeline.tokenize_chunks import tokenize_chunk
from mlm_training_pipeline.bucket_by_length import (
    compute_optimal_boundaries,
    get_bucket,
    pad_and_stack,
)
from financial_bert import FinancialBertTokenizer
from split_utils import is_val_document

CHUNK_SIZES = [128, 256, 512, 1024]

# -- Worker init / function for multiprocessing tokenization ---------------

_worker_tokenizer = None
_worker_max_length = None


def _init_tokenize_worker(model_name, max_length):
    global _worker_tokenizer, _worker_max_length
    _worker_tokenizer = FinancialBertTokenizer(model_name)
    _worker_max_length = max_length


def _tokenize_chunk_worker(chunk):
    return tokenize_chunk(_worker_tokenizer, chunk, _worker_max_length)


def main():
    parser = argparse.ArgumentParser(description="Prepare T5-style training data")
    parser.add_argument("--input_dirs", nargs="+", required=True,
                        help="Directories containing .md files")
    parser.add_argument("--output_dir", default="t5_training_data",
                        help="Output directory")
    parser.add_argument("--chunk_sizes", nargs="+", type=int, default=CHUNK_SIZES,
                        help="Target chunk sizes in estimated tokens")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Max token length (high to avoid truncation)")
    parser.add_argument("--model_name", default="answerdotai/ModernBERT-base")
    parser.add_argument("--num_buckets", type=int, default=50)
    parser.add_argument("--buckets", nargs="+", type=int, default=None,
                        help="Manual bucket upper bounds (overrides --num_buckets)")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of source files for validation")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Steps 1+2: Chunk → tokenize one scale at a time, spool to disk
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Steps 1+2: Multi-scale chunking & tokenization")
    print("=" * 60)

    num_workers = min(os.cpu_count() or 4, 4)
    tmp_dir = tempfile.mkdtemp(prefix="t5_prep_")
    tmp_path = os.path.join(tmp_dir, "tokenized.jsonl")
    total_seqs = 0
    source_file_set = set()
    all_lengths = []

    try:
        with open(tmp_path, "w", encoding="utf-8") as tmp_f, \
             mp.Pool(
                 processes=num_workers,
                 initializer=_init_tokenize_worker,
                 initargs=(args.model_name, args.max_length),
             ) as pool:
            for size in args.chunk_sizes:
                print(f"\n--- max_tokens={size} ---")

                # Chunk all input dirs at this scale (threaded — I/O bound)
                dir_jobs = [(size, d) for d in args.input_dirs]
                scale_chunks = []
                with ThreadPoolExecutor(max_workers=len(args.input_dirs)) as tpool:
                    for _, input_dir, chunks in tpool.map(
                        lambda job: (job[0], job[1], process_directory(job[1], max_tokens=job[0])),
                        dir_jobs,
                    ):
                        scale_chunks.extend(chunks)
                        print(f"  {len(chunks)} chunks from {input_dir}")

                # Tokenize and spool to disk immediately
                for item in tqdm(
                    pool.imap(_tokenize_chunk_worker, scale_chunks, chunksize=64),
                    total=len(scale_chunks),
                    desc=f"  Tokenizing (size={size})",
                    unit="chunk",
                ):
                    tmp_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    source_file_set.add(item["source_file"])
                    all_lengths.append(item["seq_length"])
                    total_seqs += 1
                del scale_chunks

        print(f"\nTokenized: {total_seqs} sequences")
        print(f"Lengths: min={min(all_lengths)}, max={max(all_lengths)}, "
              f"mean={sum(all_lengths)/len(all_lengths):.0f}")

        # --------------------------------------------------------------
        # Step 3: Holdout split by source file
        # --------------------------------------------------------------
        print("\n" + "=" * 60)
        print("Step 3: Train/val split by source file")
        print("=" * 60)

        val_files = {sf for sf in source_file_set
                     if is_val_document(sf, args.val_ratio)}
        train_count = len(source_file_set) - len(val_files)

        print(f"Source files: {len(source_file_set)} total, {len(val_files)} val, "
              f"{train_count} train")

        # --------------------------------------------------------------
        # Step 4: Bucket — stream from disk, one bucket at a time
        # --------------------------------------------------------------
        print("\n" + "=" * 60)
        print("Step 4: Bucketing")
        print("=" * 60)

        train_lengths = []
        with open(tmp_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if item["source_file"] not in val_files:
                    train_lengths.append(item["seq_length"])

        if args.buckets:
            bucket_bounds = sorted(args.buckets)
            print(f"Manual buckets: {bucket_bounds}")
        else:
            bucket_bounds = compute_optimal_boundaries(train_lengths, args.num_buckets)
            print(f"Optimal buckets ({len(bucket_bounds)}): {bucket_bounds}")

        # Spool into per-split per-bucket temp files
        bucket_tmp = {}  # (split, bound) -> file handle
        bucket_counts = defaultdict(int)
        bucket_lengths = defaultdict(list)
        for split_name in ("train", "val"):
            for bound in bucket_bounds:
                p = os.path.join(tmp_dir, f"{split_name}_{bound}.jsonl")
                bucket_tmp[(split_name, bound)] = open(p, "w", encoding="utf-8")

        with open(tmp_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, total=total_seqs, desc="Splitting & bucketing",
                             unit="seq"):
                item = json.loads(line)
                split = "val" if item["source_file"] in val_files else "train"
                bound = get_bucket(item["seq_length"], bucket_bounds)
                bucket_tmp[(split, bound)].write(line)
                bucket_counts[(split, bound)] += 1
                bucket_lengths[(split, bound)].append(item["seq_length"])

        for fh in bucket_tmp.values():
            fh.close()

        # Load each bucket independently, pad+stack, save .pt
        for split_name in ("train", "val"):
            split_dir = os.path.join(args.output_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            total = 0

            for bound in bucket_bounds:
                count = bucket_counts[(split_name, bound)]
                if count == 0:
                    continue
                total += count
                bl = bucket_lengths[(split_name, bound)]
                waste = sum(bound - l for l in bl) / (bound * count) * 100
                print(f"  {split_name} ≤{bound:>4}: {count:>6} seqs | "
                      f"len {min(bl):>3}-{max(bl):>3} | waste {waste:>4.1f}%")

                # Load this bucket from its temp file
                bp = os.path.join(tmp_dir, f"{split_name}_{bound}.jsonl")
                items = []
                with open(bp, "r", encoding="utf-8") as f:
                    for line in f:
                        items.append(json.loads(line))

                bucket_data = pad_and_stack(items, bound)
                del items
                output_path = os.path.join(split_dir, f"bucket_{bound}.pt")
                torch.save(bucket_data, output_path)
                del bucket_data
                mb = os.path.getsize(output_path) / 1e6
                print(f"           -> {output_path} ({mb:.1f} MB)")

            print(f"  {split_name} total: {total} sequences\n")

        print("Done.")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
