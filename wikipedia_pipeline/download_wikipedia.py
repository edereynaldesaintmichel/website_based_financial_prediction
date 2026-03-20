"""
Download a ~300MB subset of English Wikipedia for use as a regularization
corpus during full fine-tuning.

Streams from HuggingFace's wikimedia/wikipedia dataset. For each article:
1. Filter by minimum token count (default 4000) — ensures long, rich articles
2. Tag numbers using wikipedia_pipeline/tag_numbers.py (years are NOT tagged,
   numbers inside parentheses are NOT tagged)
3. Filter for minimum tagged <number> count (default 100)

Saves one .txt file per article into training_data/processed/wikipedia/.
Files are .txt (not .md) so the training pipeline can distinguish them
from financial documents.

Usage:
    python -m wikipedia_pipeline.download_wikipedia \
        --output_dir training_data/processed/wikipedia \
        --target_mb 300
"""
import argparse
import os
import re
import unicodedata
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from training_pipeline.tag_numbers import tag_numbers_in_text

# Rough words-to-tokens ratio (same as chunk_markdown.py)
TOKEN_RATIO = 1.35


def estimate_tokens(text: str) -> int:
    """Cheap token count estimate: word_count * ratio."""
    return int(len(text.split()) * TOKEN_RATIO)


def count_number_tags(text: str) -> int:
    """Count <number> tags in tagged text."""
    return text.count("<number>")


def slugify(title: str) -> str:
    """Convert article title to safe filename."""
    title = unicodedata.normalize("NFKD", title)
    title = re.sub(r'[^\w\s-]', '', title).strip().lower()
    return re.sub(r'[\s]+', '_', title)[:100]


def main():
    parser = argparse.ArgumentParser(
        description="Download & tag Wikipedia subset for regularization")
    parser.add_argument("--output_dir", default="training_data/processed/wikipedia",
                        help="Output directory for .txt files")
    parser.add_argument("--target_mb", type=float, default=300,
                        help="Target total text size in MB")
    parser.add_argument("--min_tokens", type=int, default=4000,
                        help="Minimum article length in estimated tokens")
    parser.add_argument("--min_numbers", type=int, default=100,
                        help="Minimum <number> tag count after tagging (0 to disable)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_bytes = args.target_mb * 1_000_000
    total_bytes = 0
    saved = 0
    skipped_short = 0
    skipped_numbers = 0

    print(f"Streaming English Wikipedia, targeting ~{args.target_mb:.0f} MB...")
    print(f"Filters: min_tokens={args.min_tokens}, min_numbers={args.min_numbers} (years & parenthesized excluded)")
    print(f"Output: {output_dir}\n")

    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )

    pbar = tqdm(ds, desc="Articles", unit="art")
    for article in pbar:
        text = article["text"]

        # Filter 1: minimum token count
        if estimate_tokens(text) < args.min_tokens:
            skipped_short += 1
            pbar.set_postfix(
                saved=saved, short=skipped_short, nums=skipped_numbers,
                mb=f"{total_bytes / 1e6:.1f}",
            )
            continue

        # Tag numbers (dates are excluded by tag_numbers logic)
        tagged_text = tag_numbers_in_text(text)

        # Filter 2: minimum number tags
        if args.min_numbers > 0 and count_number_tags(tagged_text) < args.min_numbers:
            skipped_numbers += 1
            pbar.set_postfix(
                saved=saved, short=skipped_short, nums=skipped_numbers,
                mb=f"{total_bytes / 1e6:.1f}",
            )
            continue

        # Save tagged text
        title = article.get("title", f"article_{saved}")
        filename = f"{saved:06d}_{slugify(title)}.txt"
        filepath = output_dir / filename
        filepath.write_text(tagged_text, encoding="utf-8")

        text_bytes = len(tagged_text.encode("utf-8"))
        total_bytes += text_bytes
        saved += 1

        pbar.set_postfix(
            saved=saved, short=skipped_short, nums=skipped_numbers,
            mb=f"{total_bytes / 1e6:.1f}",
        )

        if total_bytes >= target_bytes:
            break

    print(f"\nDone: {saved} articles, {total_bytes / 1e6:.1f} MB")
    print(f"Skipped {skipped_short} articles (< {args.min_tokens} tokens)")
    print(f"Skipped {skipped_numbers} articles (< {args.min_numbers} number tags)")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
