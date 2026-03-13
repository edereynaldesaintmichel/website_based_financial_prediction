"""
Step 1: Intelligent markdown chunking.

Splits markdown files into chunks that respect document structure:
- Splits on blank lines (paragraph/section boundaries)
- Never splits inside a table row or mid-sentence
- Greedily packs adjacent segments up to max token estimate
- Uses cheap word-count heuristic for token estimation

Usage:
    python -m training_pipeline.chunk_markdown \
        --input_dirs training_data/processed/sec_markdown_fixed training_data/processed/companies_house_markdown_tagged \
        --output_dir training_data/chunks \
        --max_tokens 512 \
        --preview  # optional: print chunks instead of saving
"""
import argparse
import json
import os
import re
from pathlib import Path
from typing import List

from tqdm import tqdm


TOKEN_RATIO = 1.35  # words-to-tokens ratio (conservative for financial text)


def estimate_tokens(text: str) -> int:
    """Cheap token count estimate: word_count * ratio."""
    return int(len(text.split()) * TOKEN_RATIO)


def split_into_segments(text: str) -> List[str]:
    """
    Split markdown into natural segments on blank lines,
    but keep table blocks and list blocks together.
    """
    lines = text.split("\n")
    segments = []
    current_segment_lines = []
    in_table = False

    for line in lines:
        stripped = line.strip()

        # Detect table rows (start with |)
        is_table_line = stripped.startswith("|") or re.match(r"^\|?[\s\-:|]+\|", stripped)

        if is_table_line:
            in_table = True
            current_segment_lines.append(line)
            continue

        # If we were in a table and hit a non-table line, flush the table
        if in_table and not is_table_line:
            in_table = False
            if current_segment_lines:
                segments.append("\n".join(current_segment_lines))
                current_segment_lines = []

        # Blank line = segment boundary (outside tables)
        if stripped == "":
            if current_segment_lines:
                segments.append("\n".join(current_segment_lines))
                current_segment_lines = []
        else:
            current_segment_lines.append(line)

    # Flush remaining
    if current_segment_lines:
        segments.append("\n".join(current_segment_lines))

    return [s for s in segments if s.strip()]


def pack_segments_into_chunks(segments: List[str], max_tokens: int) -> List[str]:
    """
    Greedily pack adjacent segments into chunks up to max_tokens.
    Never splits a segment; if a single segment exceeds max_tokens,
    it becomes its own chunk (will be truncated at tokenization time).
    """
    chunks = []
    current_parts = []
    current_est = 0

    for seg in segments:
        seg_est = estimate_tokens(seg)

        if current_parts and current_est + seg_est + 1 > max_tokens:
            # Flush current chunk
            chunks.append("\n\n".join(current_parts))
            current_parts = []
            current_est = 0

        current_parts.append(seg)
        current_est += seg_est + (1 if current_est > 0 else 0)  # +1 for join separator

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


def chunk_file(filepath: str, max_tokens: int) -> List[dict]:
    """Chunk a single markdown file. Returns list of chunk dicts."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        return []

    segments = split_into_segments(text)
    chunks = pack_segments_into_chunks(segments, max_tokens)

    source = os.path.basename(filepath)
    return [
        {
            "source_file": source,
            "chunk_index": i,
            "text": chunk,
            "estimated_tokens": estimate_tokens(chunk),
        }
        for i, chunk in enumerate(chunks)
    ]


def process_directory(input_dir: str, max_tokens: int) -> List[dict]:
    """Process all .md files in a directory."""
    input_path = Path(input_dir)
    md_files = sorted(input_path.glob("*.md"))
    all_chunks = []

    for fp in tqdm(md_files, desc=f"Chunking {input_path.name}", unit="file"):
        chunks = chunk_file(str(fp), max_tokens)
        all_chunks.extend(chunks)

    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="Chunk markdown files for MLM training")
    parser.add_argument("--input_dirs", nargs="+", default=[], help="Directories containing .md files")
    parser.add_argument("--output_dir", default="training_data/chunks", help="Output directory for chunks")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max estimated tokens per chunk")
    parser.add_argument("--preview", action="store_true", help="Print chunks to stdout instead of saving")
    parser.add_argument("--preview_file", type=str, default=None, help="Preview chunks for a single file")
    args = parser.parse_args()

    if args.preview_file:
        chunks = chunk_file(args.preview_file, args.max_tokens)
        for c in chunks:
            print(f"\n{'='*80}")
            print(f"Chunk {c['chunk_index']} | ~{c['estimated_tokens']} tokens | source: {c['source_file']}")
            print(f"{'='*80}")
            print(c["text"])
        print(f"\nTotal chunks: {len(chunks)}")
        return

    all_chunks = []
    for input_dir in args.input_dirs:
        print(f"Processing {input_dir}...")
        chunks = process_directory(input_dir, args.max_tokens)
        all_chunks.extend(chunks)
        print(f"  -> {len(chunks)} chunks from {input_dir}")

    if args.preview:
        for c in all_chunks[:20]:
            print(f"\n{'='*80}")
            print(f"Chunk {c['chunk_index']} | ~{c['estimated_tokens']} tokens | source: {c['source_file']}")
            print(f"{'='*80}")
            print(c["text"][:500])
        print(f"\nTotal chunks: {len(all_chunks)} (showing first 20)")
        return

    # Save chunks as JSONL
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "chunks.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(all_chunks)} chunks to {output_file}")

    # Print stats
    est_tokens = [c["estimated_tokens"] for c in all_chunks]
    print(f"Token estimates: min={min(est_tokens)}, max={max(est_tokens)}, "
          f"mean={sum(est_tokens)/len(est_tokens):.0f}, median={sorted(est_tokens)[len(est_tokens)//2]}")


if __name__ == "__main__":
    main()
