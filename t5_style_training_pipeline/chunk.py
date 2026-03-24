"""
Multi-scale chunking for T5-style CLS training.

Runs chunk_markdown at multiple token targets (128, 256, 512, 1024) so the
CLS token learns to compress varying sequence lengths.

Usage:
    python -m t5_style_training_pipeline.chunk \
        --input_dirs training_data/processed/sec_markdown_fixed \
                     training_data/processed/companies_house_markdown_tagged \
        --output_dir t5_training_data/chunks \
        --chunk_sizes 128 256 512 1024
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlm_training_pipeline.chunk_markdown import process_directory

CHUNK_SIZES = [128, 256, 512, 1024]


def main():
    parser = argparse.ArgumentParser(description="Multi-scale chunking for T5 CLS training")
    parser.add_argument("--input_dirs", nargs="+", required=True,
                        help="Directories containing .md files")
    parser.add_argument("--output_dir", default="t5_training_data/chunks",
                        help="Output directory for chunk JSONL files")
    parser.add_argument("--chunk_sizes", nargs="+", type=int, default=CHUNK_SIZES,
                        help="Target chunk sizes in estimated tokens")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for size in args.chunk_sizes:
        print(f"\n--- Chunking at max_tokens={size} ---")
        all_chunks = []
        for input_dir in args.input_dirs:
            chunks = process_directory(input_dir, max_tokens=size)
            all_chunks.extend(chunks)
            print(f"  {len(chunks)} chunks from {input_dir}")

        output_file = os.path.join(args.output_dir, f"chunks_{size}.jsonl")
        with open(output_file, "w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        est_tokens = [c["estimated_tokens"] for c in all_chunks]
        print(f"  Saved {len(all_chunks)} chunks to {output_file}")
        print(f"  Token estimates: min={min(est_tokens)}, max={max(est_tokens)}, "
              f"mean={sum(est_tokens)/len(est_tokens):.0f}")


if __name__ == "__main__":
    main()
