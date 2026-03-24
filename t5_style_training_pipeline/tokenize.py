"""
Tokenize multi-scale chunks for T5-style CLS training.

Reads each chunks_{size}.jsonl, tokenizes with FinancialBertTokenizer,
and writes tokenized_{size}.jsonl. Uses max_length=4096 to avoid
truncating any chunks (actual lengths are much shorter).

Usage:
    python -m t5_style_training_pipeline.tokenize \
        --input_dir t5_training_data/chunks \
        --output_dir t5_training_data/tokenized
"""
import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlm_training_pipeline.tokenize_chunks import tokenize_chunk
from financial_bert import FinancialBertTokenizer


def main():
    parser = argparse.ArgumentParser(description="Tokenize multi-scale chunks")
    parser.add_argument("--input_dir", default="t5_training_data/chunks")
    parser.add_argument("--output_dir", default="t5_training_data/tokenized")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Max token length (high to avoid truncation)")
    parser.add_argument("--model_name", default="answerdotai/ModernBERT-base")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = FinancialBertTokenizer(args.model_name)

    chunk_files = sorted(Path(args.input_dir).glob("chunks_*.jsonl"))
    if not chunk_files:
        print(f"No chunks_*.jsonl files found in {args.input_dir}")
        return

    for chunk_file in chunk_files:
        size_tag = chunk_file.stem.split("_", 1)[1]  # e.g., "128"
        output_file = os.path.join(args.output_dir, f"tokenized_{size_tag}.jsonl")

        with open(chunk_file, "r", encoding="utf-8") as f:
            total = sum(1 for _ in f)

        num_numbers = 0
        length_sum = 0

        with open(chunk_file, "r", encoding="utf-8") as fin, \
             open(output_file, "w", encoding="utf-8") as fout:
            for line in tqdm(fin, total=total, desc=f"Tokenizing chunks_{size_tag}", unit="chunk"):
                chunk = json.loads(line)
                tokenized = tokenize_chunk(tokenizer, chunk, args.max_length)
                fout.write(json.dumps(tokenized, ensure_ascii=False) + "\n")
                num_numbers += sum(tokenized["is_number_mask"])
                length_sum += tokenized["seq_length"]

        print(f"  -> {output_file}: {total} chunks, "
              f"avg length {length_sum / total:.1f}, "
              f"total numbers {num_numbers}")


if __name__ == "__main__":
    main()
