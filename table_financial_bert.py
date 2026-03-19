"""
TableFinancialModernBert: FinancialModernBert with 2D RoPE for HTML tables.

Tokenizer parses HTML <table> structure and computes two position-ID arrays:
    - position_ids_col: first half of RoPE rotation pairs use column positions
    - position_ids_row: second half of RoPE rotation pairs use row positions

Outside tables, both arrays hold identical sequential positions (standard RoPE).
Inside table cells, column positions reflect horizontal structure, row positions
reflect vertical structure, so the model attends within rows and columns with
appropriate distance sensitivity.

Column budgets are proportional to the widest cell's total token count (including
tags) in each column, plus overhead for <tr> tags and a small margin. All tokens
(content and tags) are sequentially spaced within their budget. Row positions are
evenly spaced. Colspan cells span the combined budget of their columns.
"""

import torch
import torch.nn as nn
import re
from typing import Dict, Any, List, Optional, Union, Tuple

from financial_bert import (
    FinancialBertTokenizer,
    FinancialModernBertConfig,
    FinancialModernBert,
)
from transformers import ModernBertForMaskedLM


# ─────────────────────────────────────────────────────────────
# HTML Table Parsing
# ─────────────────────────────────────────────────────────────

_TABLE_RE = re.compile(r'<table[^>]*>(.*?)</table>', re.DOTALL | re.IGNORECASE)
_TR_RE = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL | re.IGNORECASE)
_CELL_RE = re.compile(r'<(td|th)([^>]*)>(.*?)</\1>', re.DOTALL | re.IGNORECASE)
_COLSPAN_RE = re.compile(r'colspan\s*=\s*["\']?(\d+)', re.IGNORECASE)


def parse_tables(text: str) -> List[dict]:
    """
    Parse HTML tables in text and return their structure.

    Returns a list of dicts, one per table:
        char_start, char_end: character range of the full <table>...</table>
        rows: list of row dicts with:
            tag_start, tag_end: character range of the full <tr>...</tr>
            cells: list of cell dicts with:
                row, col, colspan,
                content_start, content_end: cell content only
                tag_start, tag_end: full <td>...</td> range
        num_rows, num_cols
    """
    tables = []
    for tm in _TABLE_RE.finditer(text):
        table_start = tm.start()
        table_html = tm.group(0)
        rows = []

        for row_idx, rm in enumerate(_TR_RE.finditer(table_html)):
            row_html = rm.group(0)
            row_abs_start = table_start + rm.start()
            cells = []
            col = 0

            for cm in _CELL_RE.finditer(row_html):
                colspan = 1
                cs = _COLSPAN_RE.search(cm.group(2))
                if cs:
                    colspan = int(cs.group(1))
                cells.append({
                    'row': row_idx,
                    'col': col,
                    'colspan': colspan,
                    'content_start': row_abs_start + cm.start(3),
                    'content_end': row_abs_start + cm.end(3),
                    'tag_start': row_abs_start + cm.start(),
                    'tag_end': row_abs_start + cm.end(),
                })
                col += colspan
            rows.append({
                'tag_start': row_abs_start + rm.start(),
                'tag_end': row_abs_start + rm.end(),
                'cells': cells,
            })

        num_cols = max(
            (sum(c['colspan'] for c in row['cells']) for row in rows), default=0
        )
        tables.append({
            'char_start': tm.start(),
            'char_end': tm.end(),
            'rows': rows,
            'num_rows': len(rows),
            'num_cols': num_cols,
        })
    return tables


def _tokens_in_range(
    offset_mapping: list, char_start: int, char_end: int
) -> List[int]:
    """Return token indices whose character span overlaps [char_start, char_end)."""
    return [
        i for i, (s, e) in enumerate(offset_mapping)
        if s < char_end and e > char_start
    ]


def compute_2d_positions(
    processed_text: str,
    offset_mapping: list,
    seq_len: int,
    margin: int = 10,
) -> Tuple[List[float], List[float]]:
    """
    Compute 2D RoPE position arrays for a tokenized sequence.

    Outside tables: both arrays are identical sequential positions.
    Inside tables: all tokens (content AND tags) are included in the
    budget and get sequentially spaced 2D positions. Column budgets
    are proportional to the widest cell's total token count (including
    <td>/<th> tags) per column, plus overhead for <tr> tags and margin.

    Returns (position_ids_col, position_ids_row), both of length seq_len.
    """
    pos_col = [float(i) for i in range(seq_len)]
    pos_row = [float(i) for i in range(seq_len)]

    tables = parse_tables(processed_text)
    if not tables:
        return pos_col, pos_row

    for table in tables:
        table_tokens = _tokens_in_range(
            offset_mapping, table['char_start'], table['char_end']
        )
        if not table_tokens:
            continue

        t_start = table_tokens[0]
        budget = len(table_tokens)
        pos_range = budget - 1  # position range [t_start, t_start + pos_range]
        num_rows = table['num_rows']
        num_cols = table['num_cols']

        if num_cols == 0 or pos_range == 0:
            continue

        # Find ALL tokens for each cell (including <td>/<th> tags)
        for row_dict in table['rows']:
            for cell in row_dict['cells']:
                cell['all_tokens'] = _tokens_in_range(
                    offset_mapping, cell['tag_start'], cell['tag_end']
                )

        # Find <tr> tag tokens per row (outside any cell)
        for row_dict in table['rows']:
            all_row_toks = set(_tokens_in_range(
                offset_mapping, row_dict['tag_start'], row_dict['tag_end']
            ))
            cell_toks = set()
            for cell in row_dict['cells']:
                cell_toks.update(cell['all_tokens'])
            row_dict['tr_tokens'] = sorted(all_row_toks - cell_toks)

        # Max total tokens per column (including tags, non-colspan only)
        col_max = {}
        for row_dict in table['rows']:
            for c in row_dict['cells']:
                if c['colspan'] == 1:
                    col_max[c['col']] = max(
                        col_max.get(c['col'], 0), len(c['all_tokens'])
                    )
        for j in range(num_cols):
            col_max.setdefault(j, 1)

        # <tr> tag overhead: max tr_tokens count across rows
        tr_overhead = max(
            (len(rd['tr_tokens']) for rd in table['rows']), default=0
        )

        total_max = sum(col_max[j] for j in range(num_cols)) + tr_overhead + margin
        total_max = max(total_max, 1)

        # Budget allocation: columns + tr overhead + margin
        col_budget = {}
        col_cumstart = {}
        # Reserve space for opening <tr> tags at the start of each row
        tr_open_budget = (tr_overhead / 2 / total_max) * pos_range
        running = tr_open_budget
        for j in range(num_cols):
            col_cumstart[j] = running
            col_budget[j] = (col_max[j] / total_max) * pos_range
            running += col_budget[j]
        tr_close_start = running

        # Assign 2D positions to each row
        for row_dict in table['rows']:
            row_idx = row_dict['cells'][0]['row'] if row_dict['cells'] else 0
            row_pos = t_start + (row_idx / max(num_rows - 1, 1)) * pos_range

            for c in row_dict['cells']:
                all_tokens = c['all_tokens']
                if not all_tokens:
                    continue
                col_j = c['col']

                # Column range for this cell
                if c['colspan'] > 1:
                    span_start = col_cumstart[col_j]
                    span_budget = sum(
                        col_budget.get(col_j + k, 0)
                        for k in range(c['colspan'])
                    )
                    n_tok = max(len(all_tokens), 1)
                else:
                    span_start = col_cumstart[col_j]
                    span_budget = col_budget[col_j]
                    n_tok = col_max[col_j]

                # Spread all tokens (tags + content) evenly across the budget
                for k, ti in enumerate(all_tokens):
                    frac = k / max(n_tok - 1, 1) if n_tok > 1 else 0.0
                    pos_col[ti] = t_start + span_start + frac * span_budget
                    pos_row[ti] = row_pos

            # <tr> tag tokens: opening before cells, closing after
            tr_toks = row_dict['tr_tokens']
            if tr_toks:
                # Split into opening (before first cell) and closing (after last)
                if row_dict['cells']:
                    first_cell_tok = min(
                        c['all_tokens'][0] for c in row_dict['cells']
                        if c['all_tokens']
                    )
                    last_cell_tok = max(
                        c['all_tokens'][-1] for c in row_dict['cells']
                        if c['all_tokens']
                    )
                    open_tr = [t for t in tr_toks if t < first_cell_tok]
                    close_tr = [t for t in tr_toks if t > last_cell_tok]
                else:
                    open_tr = tr_toks
                    close_tr = []

                # Opening <tr> tokens spread across [0, tr_open_budget)
                for k, ti in enumerate(open_tr):
                    n = max(len(open_tr), 1)
                    frac = k / max(n - 1, 1) if n > 1 else 0.0
                    pos_col[ti] = t_start + frac * tr_open_budget
                    pos_row[ti] = row_pos

                # Closing </tr> tokens spread across [tr_close_start, ...]
                tr_close_budget = (tr_overhead / 2 / total_max) * pos_range
                for k, ti in enumerate(close_tr):
                    n = max(len(close_tr), 1)
                    frac = k / max(n - 1, 1) if n > 1 else 0.0
                    pos_col[ti] = t_start + tr_close_start + frac * tr_close_budget
                    pos_row[ti] = row_pos

    return pos_col, pos_row


# ─────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────

class TableFinancialBertTokenizer(FinancialBertTokenizer):
    """
    Extends FinancialBertTokenizer with 2D RoPE position computation.

    Returns two extra fields: position_ids_col and position_ids_row.
    """

    def _tokenize_single(
        self, text: str, add_special_tokens: bool = True
    ) -> Dict[str, List]:
        # ── Number replacement (same as parent) ──
        numbers_data = []
        for match in self.number_pattern.finditer(text):
            sign_idx, log_val = self.parse_number_to_log(match.group(1))
            numbers_data.append({
                "start": match.start(), "end": match.end(),
                "sign_idx": sign_idx, "log_val": log_val,
            })

        placeholder = "§"
        processed_text = text
        offset = 0
        number_char_positions = []

        for num_info in numbers_data:
            start = num_info["start"] + offset
            end = num_info["end"] + offset
            processed_text = (
                processed_text[:start] + placeholder + processed_text[end:]
            )
            number_char_positions.append(start)
            offset += len(placeholder) - (num_info["end"] - num_info["start"])

        # ── Tokenize (keep offset_mapping for 2D position computation) ──
        encoded = self.base_tokenizer(
            processed_text,
            add_special_tokens=add_special_tokens,
            return_offsets_mapping=True,
            return_tensors=None,
        )
        input_ids = encoded["input_ids"]
        offset_mapping = encoded["offset_mapping"]

        # ── Map numbers to tokens (same as parent) ──
        number_token_indices = set()
        for char_pos in number_char_positions:
            for tok_idx, (start, end) in enumerate(offset_mapping):
                if start <= char_pos < end:
                    number_token_indices.add(tok_idx)
                    break

        result_input_ids = []
        is_number_mask = []
        number_values = []
        num_idx = 0

        for tok_idx, token_id in enumerate(input_ids):
            if tok_idx in number_token_indices:
                num_info = numbers_data[num_idx]
                num_idx += 1
                result_input_ids.append(self.number_placeholder_id)
                is_number_mask.append(1)
                number_values.append([
                    float(num_info["sign_idx"]),
                    float(num_info["log_val"]),
                ])
            else:
                result_input_ids.append(token_id)
                is_number_mask.append(0)
                number_values.append([0.0, 0.0])

        # ── 2D position computation ──
        seq_len = len(result_input_ids)
        position_ids_col, position_ids_row = compute_2d_positions(
            processed_text, offset_mapping, seq_len,
        )

        return {
            "input_ids": result_input_ids,
            "is_number_mask": is_number_mask,
            "number_values": number_values,
            "position_ids_col": position_ids_col,
            "position_ids_row": position_ids_row,
        }

    def __call__(
        self,
        text: Union[str, List[str]],
        padding: Union[bool, str] = True,
        truncation: bool = True,
        max_length: Optional[int] = 512,
        return_tensors: Optional[str] = "pt",
        add_special_tokens: bool = True,
    ) -> Dict[str, Any]:
        if isinstance(text, str):
            text = [text]

        batch_results = [
            self._tokenize_single(t, add_special_tokens) for t in text
        ]

        keys = [
            "input_ids", "is_number_mask", "number_values",
            "position_ids_col", "position_ids_row",
        ]
        batched = {k: [r[k] for r in batch_results] for k in keys}

        # Truncation
        if truncation and max_length:
            for i in range(len(text)):
                if len(batched["input_ids"][i]) > max_length:
                    for k in keys:
                        batched[k][i] = batched[k][i][:max_length]

        # Padding
        seq_lengths = [len(ids) for ids in batched["input_ids"]]
        if padding == "max_length" and max_length:
            pad_to = max_length
        elif padding in (True, "longest"):
            pad_to = max(seq_lengths)
        else:
            pad_to = None

        batch_attention_mask = []
        pad_id = self.pad_token_id or 0

        for i in range(len(text)):
            seq_len = len(batched["input_ids"][i])
            attn_mask = [1] * seq_len

            if pad_to and seq_len < pad_to:
                pad_len = pad_to - seq_len
                batched["input_ids"][i] += [pad_id] * pad_len
                attn_mask += [0] * pad_len
                batched["is_number_mask"][i] += [0] * pad_len
                batched["number_values"][i] += [[0.0, 0.0]] * pad_len
                batched["position_ids_col"][i] += [0.0] * pad_len
                batched["position_ids_row"][i] += [0.0] * pad_len

            batch_attention_mask.append(attn_mask)

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(
                    batched["input_ids"], dtype=torch.long
                ),
                "attention_mask": torch.tensor(
                    batch_attention_mask, dtype=torch.long
                ),
                "is_number_mask": torch.tensor(
                    batched["is_number_mask"], dtype=torch.float
                ),
                "number_values": torch.tensor(
                    batched["number_values"], dtype=torch.float
                ),
                "position_ids_col": torch.tensor(
                    batched["position_ids_col"], dtype=torch.float
                ),
                "position_ids_row": torch.tensor(
                    batched["position_ids_row"], dtype=torch.float
                ),
            }

        return {
            "input_ids": batched["input_ids"],
            "attention_mask": batch_attention_mask,
            "is_number_mask": batched["is_number_mask"],
            "number_values": batched["number_values"],
            "position_ids_col": batched["position_ids_col"],
            "position_ids_row": batched["position_ids_row"],
        }


# ─────────────────────────────────────────────────────────────
# 2D RoPE Wrapper
# ─────────────────────────────────────────────────────────────

class RoPE2DWrapper(nn.Module):
    """
    Wraps a rotary embedding module to support 2D table positions.

    When 2D positions are set via set_positions(), computes RoPE using:
        - First half of frequency bands → column positions
        - Second half of frequency bands → row positions

    When no 2D positions are set, delegates to the original module.
    The original module has no persistent state (inv_freq is a non-persistent
    buffer), so wrapping does not affect state_dict compatibility.
    """

    def __init__(self, original: nn.Module):
        super().__init__()
        self.original = original
        self._pos_col: Optional[torch.Tensor] = None
        self._pos_row: Optional[torch.Tensor] = None

    @property
    def inv_freq(self):
        return self.original.inv_freq

    def set_positions(self, pos_col: torch.Tensor, pos_row: torch.Tensor):
        self._pos_col = pos_col
        self._pos_row = pos_row

    def clear_positions(self):
        self._pos_col = None
        self._pos_row = None

    def forward(self, *args, **kwargs):
        if self._pos_col is None:
            return self.original(*args, **kwargs)

        inv_freq = self.original.inv_freq  # [head_dim // 2]

        # Interleave column and row across frequency bands so both
        # dimensions see the full frequency spectrum:
        #   even-indexed pairs → column positions
        #   odd-indexed pairs  → row positions
        col_freqs = torch.einsum(
            "bs,d->bsd", self._pos_col.float(), inv_freq[0::2]
        )  # [B, S, D/4]
        row_freqs = torch.einsum(
            "bs,d->bsd", self._pos_row.float(), inv_freq[1::2]
        )  # [B, S, D/4]

        # Interleave: [col_f0, row_f0, col_f1, row_f1, ...]
        freqs = torch.stack([col_freqs, row_freqs], dim=-1)
        freqs = freqs.reshape(*col_freqs.shape[:-1], -1)  # [B, S, D/2]
        emb = torch.cat([freqs, freqs], dim=-1)            # [B, S, D]

        return emb.cos(), emb.sin()


# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────

class TableFinancialModernBert(FinancialModernBert):
    """FinancialModernBert with 2D RoPE for HTML table regions."""

    config_class = FinancialModernBertConfig

    def __init__(self, config):
        super().__init__(config)
        self._wrap_rotary_embeddings()

    def _wrap_rotary_embeddings(self):
        """Find and wrap all rotary embedding modules with RoPE2DWrapper."""
        model = self.modernbert
        wrapped = False

        # Model-level rotary embedding (common in newer HF transformers)
        if hasattr(model, 'rotary_emb'):
            model.rotary_emb = RoPE2DWrapper(model.rotary_emb)
            wrapped = True

        # Per-layer rotary embeddings
        if hasattr(model, 'layers'):
            for layer in model.layers:
                attn = (
                    getattr(layer, 'attn', None)
                    or getattr(layer, 'self_attn', None)
                )
                if attn and hasattr(attn, 'rotary_emb'):
                    attn.rotary_emb = RoPE2DWrapper(attn.rotary_emb)
                    wrapped = True

        if not wrapped:
            raise RuntimeError(
                "Could not find rotary_emb in ModernBertModel. "
                "Check your transformers version."
            )

    def _get_rope_wrappers(self) -> List[RoPE2DWrapper]:
        """Collect all RoPE2DWrapper instances in the model."""
        wrappers = []
        model = self.modernbert
        if isinstance(getattr(model, 'rotary_emb', None), RoPE2DWrapper):
            wrappers.append(model.rotary_emb)
        if hasattr(model, 'layers'):
            for layer in model.layers:
                attn = (
                    getattr(layer, 'attn', None)
                    or getattr(layer, 'self_attn', None)
                )
                if attn and isinstance(
                    getattr(attn, 'rotary_emb', None), RoPE2DWrapper
                ):
                    wrappers.append(attn.rotary_emb)
        return wrappers

    def forward(
        self,
        input_ids: torch.Tensor,
        number_values: torch.Tensor,
        is_number_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids_col: Optional[torch.Tensor] = None,
        position_ids_row: Optional[torch.Tensor] = None,
        labels_text: Optional[torch.Tensor] = None,
        labels_magnitude: Optional[torch.Tensor] = None,
        labels_sign: Optional[torch.Tensor] = None,
    ):
        # Set 2D positions on all RoPE wrappers before the forward pass
        if position_ids_col is not None and position_ids_row is not None:
            for w in self._get_rope_wrappers():
                w.set_positions(position_ids_col, position_ids_row)

        try:
            return super().forward(
                input_ids=input_ids,
                number_values=number_values,
                is_number_mask=is_number_mask,
                attention_mask=attention_mask,
                labels_text=labels_text,
                labels_magnitude=labels_magnitude,
                labels_sign=labels_sign,
            )
        finally:
            # Always clear to avoid stale state
            for w in self._get_rope_wrappers():
                w.clear_positions()


def build_table_model(
    model_id: str = "answerdotai/ModernBERT-base",
    num_magnitude_bins: int = 128,
    sign_embed_dim: int = 8,
    magnitude_embed_dim: int = 64,
) -> TableFinancialModernBert:
    """Build a TableFinancialModernBert from a pretrained ModernBERT."""
    donor = ModernBertForMaskedLM.from_pretrained(model_id)

    config = FinancialModernBertConfig.from_pretrained(model_id)
    config.num_magnitude_bins = num_magnitude_bins
    config.sign_embed_dim = sign_embed_dim
    config.magnitude_embed_dim = magnitude_embed_dim

    model = TableFinancialModernBert(config)

    # strict=False because RoPE2DWrapper adds an 'original' submodule prefix,
    # but rotary_emb only has non-persistent buffers (inv_freq) so no keys
    # are actually missing or unexpected in practice.
    model.modernbert.load_state_dict(donor.model.state_dict(), strict=False)
    model.lm_head.dense.load_state_dict(donor.head.dense.state_dict())
    model.lm_head.norm.load_state_dict(donor.head.norm.state_dict())

    embedding_layer = model._get_embedding_layer()
    model.lm_head.decoder.weight = embedding_layer.weight
    model.lm_head.decoder.bias.data = donor.decoder.bias.data.clone()

    return model
