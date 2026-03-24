"""
Step 2: Tokenize chunks using FinancialBertTokenizer.

Reads chunks from JSONL, tokenizes each with the table-aware tokenizer,
and saves tokenized results ready for bucketing.

Usage:
    python -m training_pipeline.tokenize_chunks \
        --input training_data/chunks/chunks.jsonl \
        --output training_data/tokenized/tokenized.jsonl
"""
import argparse
import json
import os
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from financial_bert import FinancialBertTokenizer


def tokenize_chunk(tokenizer: FinancialBertTokenizer, chunk: dict, max_length: int = 512) -> dict:
    """Tokenize a single chunk. Returns tokenized data dict."""
    result = tokenizer(
        chunk["text"],
        padding=False,
        truncation=True,
        max_length=max_length,
        return_tensors=None,
        add_special_tokens=True,
    )

    # Result is batched (list of lists), unbatch since we have a single text
    return {
        "source_file": chunk["source_file"],
        "chunk_index": chunk["chunk_index"],
        "input_ids": result["input_ids"][0],
        "is_number_mask": result["is_number_mask"][0],
        "number_values": result["number_values"][0],
        "seq_length": len(result["input_ids"][0]),
    }


def main():
    parser = argparse.ArgumentParser(description="Tokenize chunks with FinancialBertTokenizer")
    parser.add_argument("--input", required=True, help="Input chunks JSONL file")
    parser.add_argument("--output", required=True, help="Output tokenized JSONL file")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--model_name", default="answerdotai/ModernBERT-base", help="Base tokenizer model")
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = FinancialBertTokenizer(args.model_name)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Count lines for progress bar
    with open(args.input, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    num_numbers = 0
    length_sum = 0

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=total_lines, desc="Tokenizing", unit="chunk"):
            chunk = json.loads(line)
            tokenized = tokenize_chunk(tokenizer, chunk, args.max_length)

            fout.write(json.dumps(tokenized, ensure_ascii=False) + "\n")

            num_numbers += sum(tokenized["is_number_mask"])
            length_sum += tokenized["seq_length"]

    print(f"\nDone. Tokenized {total_lines} chunks.")
    print(f"Total number tokens: {num_numbers}")
    print(f"Average sequence length: {length_sum / total_lines:.1f}")


if __name__ == "__main__":
    main()
