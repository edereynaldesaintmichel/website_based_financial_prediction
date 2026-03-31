"""
Test FinancialModernBert MLM predictions on a pre-tokenized document.

Loads a document from mlm_data/documents.pt, splits its token sequence into
chunks, masks 15% of tokens, runs inference with the trained model, then
reconstructs each chunk showing predictions inline.

Usage:
    python test_mlm_predictions.py \
        --doc_index 0 \
        --checkpoint checkpoints/mlm_full/checkpoint_epoch3/ \
        --max_chunks 5
"""
import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from financial_bert import build_model, FinancialBertTokenizer


def mask_sequence(input_ids, is_number_mask, number_values, mask_prob=0.15,
                  mask_token_id=50264, vocab_size=50264,
                  magnitude_max=12.0):
    """
    Apply MLM masking. Returns masked inputs and per-position ground-truth labels.
    Returns lists of dicts describing each masked position for later display.
    """
    seq_len = len(input_ids)
    masked_ids = list(input_ids)
    masked_number_values = list(number_values)
    masked_positions = []  # records for display

    for i in range(seq_len):
        if is_number_mask[i] == 1:
            # Number position
            if random.random() < mask_prob:
                masked_positions.append({
                    "pos": i,
                    "type": "number",
                    "original_log": number_values[i],
                })
                r = random.random()
                if r < 0.8:
                    masked_number_values[i] = magnitude_max + 1  # mask sentinel
                elif r < 0.9:
                    masked_number_values[i] = random.uniform(-12.0, magnitude_max)
        else:
            if random.random() < mask_prob:
                masked_positions.append({
                    "pos": i,
                    "type": "text",
                    "original_id": input_ids[i],
                })
                r = random.random()
                if r < 0.8:
                    masked_ids[i] = mask_token_id
                elif r < 0.9:
                    masked_ids[i] = random.randint(0, vocab_size - 1)
                # else: keep original (10%)

    return masked_ids, masked_number_values, masked_positions


def format_number(value):
    """Format a number nicely, handling large/small values."""
    if abs(value) < 0.01:
        return f"{value:.4e}"
    elif abs(value) >= 1e6:
        return f"{value:,.0f}"
    elif abs(value) >= 100:
        return f"{value:,.1f}"
    else:
        return f"{value:.2f}"


def reconstruct_chunk(tokenizer, input_ids, is_number_mask, number_values,
                      masked_positions, text_logits, magnitude_logits,
                      model_config):
    """
    Reconstruct text token-by-token with inline annotations for masked positions.
    Returns a markdown-formatted string.
    """
    # Build lookup of masked positions
    masked_lookup = {m["pos"]: m for m in masked_positions}

    # Get predictions for masked text positions
    pred_text_ids = torch.argmax(text_logits, dim=-1)  # [seq_len]

    # Special token IDs to skip entirely
    skip_ids = {tokenizer.base_tokenizer.cls_token_id, tokenizer.base_tokenizer.sep_token_id}

    def _has_alnum(s):
        return any(c.isalnum() for c in s)

    tokens_output = []
    for i in range(len(input_ids)):
        if i in masked_lookup:
            m = masked_lookup[i]
            if m["type"] == "text":
                original_token = tokenizer.decode([m["original_id"]])
                predicted_token = tokenizer.decode([pred_text_ids[i].item()])
                is_correct = m["original_id"] == pred_text_ids[i].item()
                orig_stripped = original_token.strip()
                pred_stripped = predicted_token.strip()
                leading = original_token[:len(original_token) - len(original_token.lstrip())]
                # Skip CLS/SEP; for non-alnum tokens just output the target
                if m["original_id"] in skip_ids:
                    continue
                if not _has_alnum(orig_stripped):
                    tokens_output.append(original_token)
                elif is_correct:
                    tokens_output.append(f"{leading}(**{pred_stripped}**)")
                else:
                    tokens_output.append(
                        f"{leading}(~~{pred_stripped}~~**{orig_stripped}**)"
                    )
            else:  # number
                decoded = tokenizer.decode_number(
                    magnitude_logits[i], model_config
                )
                pred_log = decoded["log_magnitude"]
                pred_value = tokenizer.log_to_value(pred_log)
                orig_value = tokenizer.log_to_value(m["original_log"])
                tokens_output.append(
                    f"(~~{format_number(pred_value)}~~**{format_number(orig_value)}**)"
                )
        else:
            # Skip CLS/SEP
            if input_ids[i] in skip_ids:
                continue
            if is_number_mask[i] == 1:
                value = tokenizer.log_to_value(number_values[i])
                tokens_output.append(format_number(value))
            else:
                tokens_output.append(tokenizer.decode([input_ids[i]]))

    return "".join(tokens_output)


def main():
    parser = argparse.ArgumentParser(description="Test MLM predictions on a pre-tokenized document")
    parser.add_argument("--documents", default="mlm_data/documents.pt",
                        help="Path to documents.pt")
    parser.add_argument("--doc_index", type=int, default=None,
                        help="Document index in documents.pt (default: random)")
    parser.add_argument("--source_file", type=str, default=None,
                        help="Select document by source_file name (e.g. '1750_2018-07-11.md')")
    parser.add_argument("--checkpoint", default="checkpoints/mlm_full_baseline/checkpoint_epoch2",
                        help="Path to checkpoint directory (contains full_model.pt)")
    parser.add_argument("--model_name", default="answerdotai/ModernBERT-base")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens per chunk")
    parser.add_argument("--max_chunks", type=int, default=0,
                        help="Max chunks to process (0 = all)")
    parser.add_argument("--mask_prob", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="Output .md file (default: <source_file>_predictions.md)")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "mps" if torch.backends.mps.is_available()
                              else "cpu")
    print(f"Device: {device}")

    # 1. Load documents and select one
    print(f"\n### Loading {args.documents}...")
    documents = torch.load(args.documents, weights_only=False)
    print(f"  {len(documents)} documents")

    if args.source_file:
        doc = next((d for d in documents if d["source_file"] == args.source_file), None)
        if doc is None:
            print(f"Error: source_file '{args.source_file}' not found")
            sys.exit(1)
    elif args.doc_index is not None:
        doc = documents[args.doc_index]
    else:
        doc = random.choice(documents)

    source_name = doc["source_file"]
    print(f"  Selected: {source_name} ({doc['seq_length']} tokens)")

    # 2. Chunk the pre-tokenized sequence
    input_ids = doc["input_ids"].tolist()
    is_number_mask = doc["is_number_mask"].tolist()
    number_values = doc["number_values"].tolist()
    seq_len = doc["seq_length"]
    chunk_size = args.max_tokens

    tokenized_chunks = []
    for i in range(0, seq_len, chunk_size):
        end = min(i + chunk_size, seq_len)
        tokenized_chunks.append({
            "input_ids": input_ids[i:end],
            "is_number_mask": is_number_mask[i:end],
            "number_values": number_values[i:end],
            "chunk_index": len(tokenized_chunks),
        })

    print(f"  {len(tokenized_chunks)} chunks of up to {chunk_size} tokens")

    if args.max_chunks > 0:
        tokenized_chunks = tokenized_chunks[:args.max_chunks]

    # 3. Load tokenizer
    print("### Loading tokenizer...")
    tokenizer = FinancialBertTokenizer(args.model_name)

    # 3. Build model and load full checkpoint
    print("### Building model...")
    model = build_model(args.model_name)

    ckpt_file = os.path.join(args.checkpoint, "full_model.pt")
    print(f"### Loading checkpoint from {ckpt_file}...")
    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    # Strip _orig_mod. prefix from torch.compile'd checkpoints
    state_dict = ckpt["model_state_dict"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    model_config = model.config

    # Output file path
    if args.output:
        output_path = args.output
    else:
        stem = os.path.splitext(source_name)[0]
        output_path = f"{stem}_predictions.md"

    # 4. Process each chunk
    print(f"\n### Running inference on {len(tokenized_chunks)} chunks...")

    total_text_masked = 0
    total_text_correct = 0
    total_num_masked = 0
    total_num_close = 0

    md_lines = []
    md_lines.append(f"# MLM Predictions: `{source_name}`\n")
    md_lines.append(f"| Setting | Value |")
    md_lines.append(f"|---|---|")
    md_lines.append(f"| Checkpoint | `{args.checkpoint}` |")
    md_lines.append(f"| Mask probability | {args.mask_prob} |")
    md_lines.append(f"| Seed | {args.seed} |")
    md_lines.append(f"| Chunks processed | {len(tokenized_chunks)} |")
    md_lines.append(f"| Device | {device} |")
    md_lines.append("")
    md_lines.append("> **Legend:** "
                    "(**token**) = correct prediction &nbsp;|&nbsp; "
                    "(~~wrong~~**right**) = wrong prediction &nbsp;|&nbsp; "
                    "plain text = not masked")
    md_lines.append("")

    from tqdm import tqdm
    for tc in tqdm(tokenized_chunks, desc="Inference", unit="chunk"):
        input_ids = tc["input_ids"]
        is_number_mask = tc["is_number_mask"]
        number_values = tc["number_values"]
        seq_len = len(input_ids)

        # Mask
        masked_ids, masked_nv, masked_positions = mask_sequence(
            input_ids, is_number_mask, number_values,
            mask_prob=args.mask_prob,
            mask_token_id=tokenizer.mask_token_id,
            magnitude_max=model_config.magnitude_max,
        )

        # Prepare tensors (batch size 1)
        t_input_ids = torch.tensor([masked_ids], dtype=torch.long, device=device)
        t_attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)
        t_is_number_mask = torch.tensor([is_number_mask], dtype=torch.float, device=device)
        t_number_values = torch.tensor([masked_nv], dtype=torch.float, device=device)

        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=t_input_ids,
                attention_mask=t_attention_mask,
                is_number_mask=t_is_number_mask,
                number_values=t_number_values,
            )

        text_logits = outputs["text_logits"][0].cpu()      # [seq_len, vocab]
        mag_logits = outputs["magnitude_logits"][0].cpu()   # [seq_len, bins]

        # Compute accuracy stats
        pred_text_ids = torch.argmax(text_logits, dim=-1)
        chunk_text_masked = 0
        chunk_text_correct = 0
        chunk_num_masked = 0
        chunk_num_close = 0
        for m in masked_positions:
            if m["type"] == "text":
                total_text_masked += 1
                chunk_text_masked += 1
                if m["original_id"] == pred_text_ids[m["pos"]].item():
                    total_text_correct += 1
                    chunk_text_correct += 1
            else:
                total_num_masked += 1
                chunk_num_masked += 1
                decoded = tokenizer.decode_number(
                    mag_logits[m["pos"]], model_config, k=1
                )
                if abs(m["original_log"] - decoded["log_magnitude"]) < 1.0:
                    total_num_close += 1
                    chunk_num_close += 1

        # Reconstruct
        reconstructed = reconstruct_chunk(
            tokenizer, input_ids, is_number_mask, number_values,
            masked_positions, text_logits, mag_logits,
            model_config,
        )

        # Per-chunk accuracy string
        acc_parts = []
        if chunk_text_masked > 0:
            pct = 100 * chunk_text_correct / chunk_text_masked
            acc_parts.append(f"text {chunk_text_correct}/{chunk_text_masked} ({pct:.0f}%)")
        if chunk_num_masked > 0:
            pct = 100 * chunk_num_close / chunk_num_masked
            acc_parts.append(f"numbers {chunk_num_close}/{chunk_num_masked} ({pct:.0f}%)")
        acc_str = " · ".join(acc_parts) if acc_parts else "no masks"

        md_lines.append(f"---\n")
        md_lines.append(f"## Chunk {tc['chunk_index']}  ")
        md_lines.append(f"*{seq_len} tokens · {acc_str}*\n")
        md_lines.append(reconstructed)
        md_lines.append("")

    # Summary section
    md_lines.append("---\n")
    md_lines.append("## Summary\n")
    if total_text_masked > 0:
        text_pct = 100 * total_text_correct / total_text_masked
        md_lines.append(f"| Metric | Value |")
        md_lines.append(f"|---|---|")
        md_lines.append(f"| Text tokens correct | {total_text_correct} / {total_text_masked} ({text_pct:.1f}%) |")
    if total_num_masked > 0:
        num_pct = 100 * total_num_close / total_num_masked
        md_lines.append(f"| Numbers within 1 OOM | {total_num_close} / {total_num_masked} ({num_pct:.1f}%) |")
    md_lines.append(f"| Total masked positions | {total_text_masked + total_num_masked} |")
    md_lines.append("")

    # Write the file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\nSaved to {output_path}")

    # Print summary to console too
    if total_text_masked > 0:
        print(f"  Text: {total_text_correct}/{total_text_masked} "
              f"({100*total_text_correct/total_text_masked:.1f}%)")
    if total_num_masked > 0:
        print(f"  Numbers: {total_num_close}/{total_num_masked} "
              f"({100*total_num_close/total_num_masked:.1f}%)")
    print(f"  Total masked: {total_text_masked + total_num_masked}")


if __name__ == "__main__":
    main()
