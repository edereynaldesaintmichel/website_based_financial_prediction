"""Smoke test: verify 2D RoPE model matches vanilla ModernBERT on text-only tokens."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import ModernBertForMaskedLM
from table_financial_bert import (
    TableFinancialBertTokenizer, build_table_model, RoPE2DWrapper,
    ROW_DIRECTION_UNIT_VECT, COLUMN_DIRECTION_UNIT_VECT, D_RC, D_COL,
)

SEED = 42
MASK_PROB = 0.10
# First number token index — mask only text tokens before this
FIRST_NUMBER_POS = 91


def main():
    torch.manual_seed(SEED)
    print(f"Constants: ROW_DIR={ROW_DIRECTION_UNIT_VECT}, COL_DIR={COLUMN_DIRECTION_UNIT_VECT}, D_RC={D_RC}, D_COL={D_COL}")

    # --- 1. Tokenize ---
    tokenizer = TableFinancialBertTokenizer()
    input_path = os.path.join(os.path.dirname(__file__), "tokenization_output.md")
    with open(input_path) as f:
        text = f.read()

    enc = tokenizer(text, truncation=True, max_length=512, padding=False,
                    return_tensors="pt")
    full_seq_len = enc["input_ids"].shape[1]
    print(f"Full sequence: {full_seq_len} tokens")

    # Truncate to first FIRST_NUMBER_POS tokens (all text, pos_col == pos_row)
    T = FIRST_NUMBER_POS
    input_ids = enc["input_ids"][:, :T].clone()
    attention_mask = enc["attention_mask"][:, :T].clone()
    pos_col = enc["position_ids_col"][:, :T].clone()
    pos_row = enc["position_ids_row"][:, :T].clone()
    number_values = enc["number_values"][:, :T].clone()
    is_number_mask = enc["is_number_mask"][:, :T].clone()

    # Verify all tokens are text (pos_col == pos_row, is_number_mask == 0)
    assert (pos_col == pos_row).all(), "Some tokens have 2D positions in truncated range!"
    assert (is_number_mask == 0).all(), "Some tokens are numbers in truncated range!"
    print(f"Truncated to {T} text-only tokens (pos_col == pos_row everywhere)")

    # --- 2. MLM masking (reproducible) ---
    torch.manual_seed(SEED)
    rand = torch.rand(input_ids.shape)
    rand[:, 0] = 0   # don't mask CLS
    rand[:, -1] = 0   # don't mask SEP (if it's the last token)
    mask_positions = rand < MASK_PROB
    n_masked = mask_positions.sum().item()

    labels = input_ids.clone()
    labels[~mask_positions] = -100
    masked_input_ids = input_ids.clone()
    masked_input_ids[mask_positions] = tokenizer.mask_token_id
    print(f"Masked {n_masked}/{T} tokens ({100*n_masked/T:.1f}%)")

    # Show a few masked positions
    idxs = mask_positions[0].nonzero(as_tuple=True)[0][:8].tolist()
    base_tok = tokenizer.base_tokenizer
    print(f"  Sample masked indices: {idxs}")
    for idx in idxs:
        print(f"    [{idx}] '{base_tok.decode([input_ids[0, idx].item()])}'")

    # --- 3. Load models ---
    print("\nLoading pretrained ModernBERT...")
    model_id = "answerdotai/ModernBERT-base"

    # Vanilla ModernBERT
    vanilla = ModernBertForMaskedLM.from_pretrained(model_id)
    vanilla.eval()

    # Our 2D RoPE model (same pretrained weights)
    table_model = build_table_model(model_id)
    table_model.eval()

    print("Both models loaded.")

    # --- 4. Forward pass: vanilla ---
    print("\n--- Vanilla ModernBERT ---")
    with torch.no_grad():
        vanilla_out = vanilla(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
    vanilla_loss = vanilla_out.loss.item()
    print(f"  Loss: {vanilla_loss:.6f}")

    # --- 5. Forward pass: our model ---
    print("\n--- TableFinancialModernBert (2D RoPE) ---")
    # For text-only tokens: labels_sign and labels_magnitude are all -100
    labels_sign = torch.full_like(labels, -100)
    labels_magnitude = torch.full(labels.shape, -100.0)

    with torch.no_grad():
        table_out = table_model(
            input_ids=masked_input_ids,
            number_values=number_values,
            is_number_mask=is_number_mask,
            attention_mask=attention_mask,
            position_ids_col=pos_col,
            position_ids_row=pos_row,
            labels_text=labels,
            labels_sign=labels_sign,
            labels_magnitude=labels_magnitude,
        )
    table_loss = table_out["loss_text"].item()
    print(f"  Loss (text): {table_loss:.6f}")

    # --- 6. Compare ---
    diff = abs(vanilla_loss - table_loss)
    print(f"\n{'=' * 50}")
    print(f"Vanilla loss:    {vanilla_loss:.6f}")
    print(f"2D RoPE loss:    {table_loss:.6f}")
    print(f"Absolute diff:   {diff:.2e}")
    print(f"Match: {'OK' if diff < 1e-4 else 'FAIL'}")
    print(f"{'=' * 50}")

    # --- 7. Also compare logits directly ---
    vanilla_logits = vanilla_out.logits  # [1, T, vocab]
    table_logits = table_out["text_logits"]  # [1, T, vocab]
    logit_diff = (vanilla_logits - table_logits).abs()
    print(f"\nLogit comparison:")
    print(f"  Max abs diff:  {logit_diff.max().item():.2e}")
    print(f"  Mean abs diff: {logit_diff.mean().item():.2e}")

    # Check per-position max diff
    per_pos_max = logit_diff[0].max(dim=-1).values  # [T]
    worst_pos = per_pos_max.argmax().item()
    print(f"  Worst position: {worst_pos} (max diff={per_pos_max[worst_pos].item():.2e})")

    print("\nDone.")


if __name__ == "__main__":
    main()
