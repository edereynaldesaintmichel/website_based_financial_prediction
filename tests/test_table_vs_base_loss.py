"""
Compare TableFinancialModernBert vs FinancialModernBert losses.
1. Plain text (no tables): losses should be exactly identical (zero-init embeddings).
2. Text with tables: losses can differ (different tokenization), but should be in the same ballpark.
"""
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from financial_bert import FinancialBertTokenizer, build_model
from table_financial_bert import TableFinancialBertTokenizer, build_table_model

MODEL_ID = "answerdotai/ModernBERT-base"

def make_mlm_labels(input_ids, attention_mask, mask_token_id, mask_prob=0.15):
    """Create MLM labels: mask 15% of tokens, return masked input_ids and label tensors."""
    labels = input_ids.clone()
    rand = torch.rand(input_ids.shape)
    mask = (rand < mask_prob) & (attention_mask == 1)
    # Don't mask special tokens (first and last)
    mask[:, 0] = False
    mask[:, -1] = False
    labels[~mask] = -100
    masked_ids = input_ids.clone()
    masked_ids[mask] = mask_token_id
    return masked_ids, labels


def test_plain_text():
    """Plain text: both models should produce identical loss."""
    print("=" * 60)
    print("TEST 1: Plain text (no tables)")
    print("=" * 60)

    base_tok = FinancialBertTokenizer(MODEL_ID)
    table_tok = TableFinancialBertTokenizer(MODEL_ID)

    base_model = build_model(MODEL_ID)
    table_model = build_table_model(MODEL_ID)

    # Sync randomly-initialized heads so only architecture matters
    table_model.number_embedder.load_state_dict(base_model.number_embedder.state_dict())
    table_model.number_head.load_state_dict(base_model.number_head.state_dict())

    base_model.eval()
    table_model.eval()

    text = "The company reported earnings of <number>42.5</number> million in Q3, up from <number>38.1</number> million last year."

    base_batch = base_tok(text, max_length=64)
    table_batch = table_tok(text, max_length=64)

    # Verify tokenization is identical for plain text
    assert torch.equal(base_batch["input_ids"], table_batch["input_ids"]), "input_ids differ on plain text!"
    assert torch.equal(base_batch["is_number_mask"], table_batch["is_number_mask"]), "is_number_mask differs!"
    assert torch.equal(base_batch["number_values"], table_batch["number_values"]), "number_values differ!"
    print("  Tokenization: identical ✓")

    # Create MLM labels
    torch.manual_seed(42)
    masked_ids_base, labels = make_mlm_labels(
        base_batch["input_ids"], base_batch["attention_mask"],
        base_tok.mask_token_id
    )
    torch.manual_seed(42)
    masked_ids_table, labels_table = make_mlm_labels(
        table_batch["input_ids"], table_batch["attention_mask"],
        table_tok.mask_token_id
    )

    # magnitude labels (use the actual number values for masked number positions)
    labels_magnitude = torch.full_like(base_batch["number_values"], -100.0)
    num_mask = base_batch["is_number_mask"] == 1
    labels_magnitude[num_mask] = base_batch["number_values"][num_mask]

    with torch.no_grad():
        base_out = base_model(
            input_ids=masked_ids_base,
            number_values=base_batch["number_values"],
            is_number_mask=base_batch["is_number_mask"],
            attention_mask=base_batch["attention_mask"],
            labels_text=labels,
            labels_magnitude=labels_magnitude,
        )
        table_out = table_model(
            input_ids=masked_ids_table,
            number_values=table_batch["number_values"],
            is_number_mask=table_batch["is_number_mask"],
            attention_mask=table_batch["attention_mask"],
            table_row_index=table_batch["table_row_index"],
            table_col_index=table_batch["table_col_index"],
            table_mask=table_batch["table_mask"],
            table_num_rows=table_batch["table_num_rows"],
            table_num_cols=table_batch["table_num_cols"],
            labels_text=labels_table,
            labels_magnitude=labels_magnitude,
        )

    base_loss = base_out["loss"].item()
    table_loss = table_out["loss"].item()
    diff = abs(base_loss - table_loss)

    print(f"  Base model loss:       {base_loss:.6f}  (text: {base_out['loss_text'].item():.4f}, mag: {base_out['loss_mag'].item():.4f})")
    print(f"  Table model loss:      {table_loss:.6f}  (text: {table_out['loss_text'].item():.4f}, mag: {table_out['loss_mag'].item():.4f})")
    print(f"  Absolute diff (total): {diff:.2e}")

    assert diff < 1e-5, f"Losses differ by {diff} on plain text (expected identical)!"
    print("  Losses identical ✓")


def test_table_text_same_tokenizer():
    """Text with a table, same tokenizer: losses should be exactly identical (zero-init)."""
    print()
    print("=" * 60)
    print("TEST 2: Text with HTML table (same tokenizer)")
    print("=" * 60)

    table_tok = TableFinancialBertTokenizer(MODEL_ID)

    base_model = build_model(MODEL_ID)
    table_model = build_table_model(MODEL_ID)

    # Sync randomly-initialized heads
    table_model.number_embedder.load_state_dict(base_model.number_embedder.state_dict())
    table_model.number_head.load_state_dict(base_model.number_head.state_dict())

    base_model.eval()
    table_model.eval()

    text = """Revenue breakdown:
<table><tr><th>Segment</th><th>Revenue</th></tr>
<tr><td>Cloud</td><td><number>1250.3</number></td></tr>
<tr><td>Services</td><td><number>840.7</number></td></tr></table>
Total revenue was <number>2091.0</number> million."""

    batch = table_tok(text, max_length=128)
    print(f"  Sequence length: {batch['input_ids'].shape[1]}")

    # Create MLM labels
    torch.manual_seed(42)
    masked_ids, labels = make_mlm_labels(
        batch["input_ids"], batch["attention_mask"],
        table_tok.mask_token_id
    )
    labels_mag = torch.full_like(batch["number_values"], -100.0)
    num_mask = batch["is_number_mask"] == 1
    labels_mag[num_mask] = batch["number_values"][num_mask]

    with torch.no_grad():
        base_out = base_model(
            input_ids=masked_ids,
            number_values=batch["number_values"],
            is_number_mask=batch["is_number_mask"],
            attention_mask=batch["attention_mask"],
            labels_text=labels,
            labels_magnitude=labels_mag,
        )
        table_out = table_model(
            input_ids=masked_ids,
            number_values=batch["number_values"],
            is_number_mask=batch["is_number_mask"],
            attention_mask=batch["attention_mask"],
            table_row_index=batch["table_row_index"],
            table_col_index=batch["table_col_index"],
            table_mask=batch["table_mask"],
            table_num_rows=batch["table_num_rows"],
            table_num_cols=batch["table_num_cols"],
            labels_text=labels,
            labels_magnitude=labels_mag,
        )

    base_loss = base_out["loss"].item()
    table_loss = table_out["loss"].item()
    diff = abs(base_loss - table_loss)

    print(f"  Base model loss:       {base_loss:.6f}  (text: {base_out['loss_text'].item():.4f}, mag: {base_out['loss_mag'].item():.4f})")
    print(f"  Table model loss:      {table_loss:.6f}  (text: {table_out['loss_text'].item():.4f}, mag: {table_out['loss_mag'].item():.4f})")
    print(f"  Absolute diff (total): {diff:.2e}")

    assert diff < 1e-5, f"Losses differ by {diff} on table text with same tokenizer (expected identical)!"
    print("  Losses identical ✓")


if __name__ == "__main__":
    test_plain_text()
    test_table_text_same_tokenizer()
    print("\n✅ All tests passed!")
