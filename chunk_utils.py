"""
Shared chunking and batching utilities for T5 and CLS aggregator pipelines.

Functions in this module are used by:
  - t5_style_training_pipeline/train.py
  - cls_aggregator_training_pipeline/train.py
"""
import bisect
import torch

from financial_bert import FinancialBertTokenizer

MIN_TAIL_CHUNK = 16  # absorb trailing chunks shorter than this


# ---------------------------------------------------------------------------
# Token info
# ---------------------------------------------------------------------------

def get_token_info(pretrained_id):
    """Extract special token IDs needed for training."""
    tokenizer = FinancialBertTokenizer(pretrained_id)
    base = tokenizer.base_tokenizer
    newline_ids = set(base.encode("\n", add_special_tokens=False))
    info = {
        "cls_id": base.cls_token_id,
        "sep_id": base.sep_token_id,
        "pad_id": base.pad_token_id,
        "mask_id": tokenizer.mask_token_id,
        "newline_ids": newline_ids,
    }
    del tokenizer
    return info


# ---------------------------------------------------------------------------
# Boundary detection
# ---------------------------------------------------------------------------

def get_boundaries(input_ids, newline_ids):
    """Find chunk boundary candidates (positions after newline tokens).

    Works in content space (CLS/SEP stripped: position 0 = first content token).
    Returns sorted list of boundary positions.
    """
    content = input_ids[1:-1]  # strip CLS at 0 and SEP at -1
    mask = torch.zeros(len(content), dtype=torch.bool)
    for nl_id in newline_ids:
        mask |= (content == nl_id)
    # Boundary = position AFTER the newline (i.e. start of next chunk)
    return (mask.nonzero(as_tuple=False).squeeze(-1) + 1).tolist()


def snap_to_boundary(target, boundaries, min_pos):
    """Snap a target position to the nearest boundary candidate."""
    if not boundaries:
        return target
    idx = bisect.bisect_left(boundaries, target)
    candidates = []
    if idx > 0 and boundaries[idx - 1] > min_pos:
        candidates.append(boundaries[idx - 1])
    if idx < len(boundaries):
        candidates.append(boundaries[idx])
    if not candidates:
        return target
    return min(candidates, key=lambda b: abs(b - target))


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_spans(content_len, boundaries, size_fn):
    """Generate (start, end) chunk spans for a document's content."""
    spans = []
    pos = 0
    while pos < content_len:
        target = size_fn()
        raw_end = min(pos + target, content_len)

        if raw_end < content_len:
            end = snap_to_boundary(raw_end, boundaries, pos)
            if end <= pos:
                end = min(pos + target, content_len)
            # Absorb tiny tail
            if content_len - end < MIN_TAIL_CHUNK:
                end = content_len
        else:
            end = content_len

        spans.append((pos, end))
        pos = end
    return spans


def extract_chunk(doc, start, end, cls_id, sep_id):
    """Extract a chunk from document content (content space), wrap with CLS/SEP."""
    offset = 1 + start  # +1 to skip document CLS
    length = end - start

    c_ids = doc["input_ids"][offset:offset + length]
    c_mask = doc["is_number_mask"][offset:offset + length]
    c_vals = doc["number_values"][offset:offset + length]

    chunk_ids = torch.cat([
        torch.tensor([cls_id], dtype=c_ids.dtype), c_ids,
        torch.tensor([sep_id], dtype=c_ids.dtype),
    ])
    chunk_mask = torch.cat([
        torch.tensor([False]), c_mask.bool(),
        torch.tensor([False]),
    ])
    chunk_vals = torch.cat([
        torch.tensor([0.0]), c_vals.float(),
        torch.tensor([0.0]),
    ])

    return {
        "input_ids": chunk_ids,
        "is_number_mask": chunk_mask,
        "number_values": chunk_vals,
        "seq_length": len(chunk_ids),
        "doc_start": start,
        "doc_end": end,
    }


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def pad_and_collate(chunks, pad_id):
    """Pad and stack chunk dicts into batch tensors."""
    max_len = max(c["seq_length"] for c in chunks)
    B = len(chunks)

    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    is_number_mask = torch.zeros(B, max_len, dtype=torch.long)
    number_values = torch.zeros(B, max_len, dtype=torch.float32)
    attention_mask = torch.zeros(B, max_len, dtype=torch.long)

    for i, c in enumerate(chunks):
        l = c["seq_length"]
        input_ids[i, :l] = c["input_ids"]
        is_number_mask[i, :l] = c["is_number_mask"].long()
        number_values[i, :l] = c["number_values"].float()
        attention_mask[i, :l] = 1

    return input_ids, is_number_mask, number_values, attention_mask
