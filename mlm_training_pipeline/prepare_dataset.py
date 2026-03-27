"""
Tokenize whole documents for MLM training.

Each document is tokenized as a single long sequence (no chunking, no truncation).
Chunking happens at the start of each training epoch, not here.

Includes both financial documents (.md) and Wikipedia regularization documents (.txt).

Output: a single .pt file containing a list of dicts, one per document:
    {
        "source_file":    str,
        "input_ids":      LongTensor  (S,),
        "is_number_mask": BoolTensor  (S,),
        "number_values":  FloatTensor (S,),
        "seq_length":     int,
    }

Usage:
    python -m mlm_training_pipeline.prepare_dataset \
        --input_dirs training_data/processed/SEC_10k_markdown_tagged \
                     training_data/processed/companies_house_markdown_tagged \
                     training_data/processed/wikipedia_tagged \
        --output mlm_data/documents.pt
"""
import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

# Prevent HuggingFace tokenizers (Rust rayon) from spawning hundreds of threads per worker
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from financial_bert import FinancialBertTokenizer

MAX_LENGTH = 128_000  # generous ceiling; no real truncation expected

# -- Multiprocessing workers --------------------------------------------------

_worker_tokenizer = None
_worker_max_length = None


def _init_worker(model_name, max_length):
    global _worker_tokenizer, _worker_max_length
    _worker_tokenizer = FinancialBertTokenizer(model_name)
    _worker_max_length = max_length


def _tokenize_file(args):
    filepath, source_file = args
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        return None

    result = _worker_tokenizer(
        text,
        padding=False,
        truncation=True,
        max_length=_worker_max_length,
        return_tensors=None,
        add_special_tokens=True,
    )

    input_ids = result["input_ids"][0]
    is_number_mask = result["is_number_mask"][0]
    number_values = result["number_values"][0]

    # Return plain lists to avoid torch tensor fd-sharing issues in multiprocessing
    return {
        "source_file": source_file,
        "input_ids": input_ids,
        "is_number_mask": is_number_mask,
        "number_values": number_values,
        "seq_length": len(input_ids),
    }


def main():
    parser = argparse.ArgumentParser(description="Tokenize whole documents for MLM training")
    parser.add_argument("--input_dirs", nargs="+", required=True,
                        help="Directories containing .md/.txt files")
    parser.add_argument("--output", default="mlm_data/documents.pt",
                        help="Output .pt file")
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH,
                        help="Max token length (safety ceiling)")
    parser.add_argument("--model_name", default="answerdotai/ModernBERT-base")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of tokenization workers (default: cpu_count)")
    args = parser.parse_args()

    # Collect all files
    all_files = []  # (full_path, source_file_name)
    for input_dir in args.input_dirs:
        p = Path(input_dir)
        for ext in ("*.md", "*.txt"):
            for fp in sorted(p.glob(ext)):
                all_files.append((str(fp), fp.name))

    print(f"Found {len(all_files)} files across {len(args.input_dirs)} directories")

    # Tokenize in parallel
    num_workers = args.num_workers or min(max((os.cpu_count() or 4) - 2, 1), 32)
    documents = []
    total_tokens = 0
    lengths = []

    with mp.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(args.model_name, args.max_length),
    ) as pool:
        for result in tqdm(
            pool.imap(_tokenize_file, all_files, chunksize=32),
            total=len(all_files),
            desc="Tokenizing documents",
            unit="doc",
        ):
            if result is not None:
                documents.append(result)
                total_tokens += result["seq_length"]
                lengths.append(result["seq_length"])

    # Convert to tensors in the main process (avoids fd-sharing across workers)
    for doc in documents:
        doc["input_ids"] = torch.tensor(doc["input_ids"], dtype=torch.long)
        doc["is_number_mask"] = torch.tensor(doc["is_number_mask"], dtype=torch.bool)
        doc["number_values"] = torch.tensor(doc["number_values"], dtype=torch.float32)

    # Sort by source_file for reproducibility
    documents.sort(key=lambda d: d["source_file"])

    # Stats
    n_financial = sum(1 for d in documents if d["source_file"].endswith(".md"))
    n_regularization = sum(1 for d in documents if d["source_file"].endswith(".txt"))
    lengths.sort()
    print(f"\nTokenized {len(documents)} documents ({len(all_files) - len(documents)} empty/skipped)")
    print(f"  Financial (.md): {n_financial}")
    print(f"  Regularization (.txt): {n_regularization}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Lengths: min={lengths[0]}, median={lengths[len(lengths)//2]}, "
          f"max={lengths[-1]}, mean={total_tokens/len(documents):.0f}")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(documents, args.output)
    mb = os.path.getsize(args.output) / 1e6
    print(f"\nSaved to {args.output} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
