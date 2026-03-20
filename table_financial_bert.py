"""
TableFinancialModernBert: FinancialModernBert with 2D RoPE for HTML tables.

Tokenizer parses HTML <table> structure and computes two position-ID arrays:
    - position_ids_col (axis 1, even RoPE pairs): column/horizontal positions
    - position_ids_row (axis 2, odd RoPE pairs): row/vertical positions

Outside tables, both arrays hold identical sequential positions (standard RoPE).
Inside tables, positions follow a column-aligned 2D layout:
    - Within cells: (+1, +1) per token (vanilla RoPE preserved)
    - Columns are aligned: each column occupies max_tokens_in_col + D_RC
      along ROW_DIRECTION_UNIT_VECT, so same-column cells start at the
      same position regardless of neighboring cell sizes.
    - Row-to-row: offset of D_COL along COLUMN_DIRECTION_UNIT_VECT per row
"""

import torch
import torch.nn as nn
import re
from typing import Dict, Any, List, Optional, Union, Tuple
from bs4 import BeautifulSoup

from financial_bert import (
    FinancialBertTokenizer,
    FinancialModernBertConfig,
    FinancialModernBert,
)
from transformers import ModernBertForMaskedLM


_TABLE_RE = re.compile(r'<table[^>]*>.*?</table>', re.DOTALL | re.IGNORECASE)

# Structural tokens for table boundaries and cell delimiters.
# [unused0] and [unused1] are pre-allocated in ModernBERT's vocabulary
# with random embeddings — they'll learn table semantics during fine-tuning.
TABLE_START_ID = 50285  # [unused0]
TABLE_END_ID = 50286    # [unused1]
TAB_ID = 186            # \t — cell delimiter

# Direction vectors for 2D RoPE, as (col_axis_component, row_axis_component).
# Row direction: primary axis for horizontal cell-to-cell traversal.
# Column direction: primary axis for vertical row-to-row traversal.
# Symmetric by construction: swapping components mirrors the vector.
ROW_DIRECTION_UNIT_VECT = (1.0, 1)
COLUMN_DIRECTION_UNIT_VECT = (0.3, 1.0)

# Inter-cell gap within a row, measured along ROW_DIRECTION_UNIT_VECT.
D_RC = 4
# Inter-row spacing, measured along COLUMN_DIRECTION_UNIT_VECT.
D_COL = 8


def parse_table_grid(table_html: str) -> List[List[Tuple[int, str]]]:
    """Parse an HTML table into a grid of (col_index, cell_content) per row.

    Handles colspan and rowspan. Cell content preserves inner HTML (including
    <number> tags) but strips whitespace. Spanned cells are omitted.

    Returns: grid[row] = [(col_idx, content_html), ...]
    """
    soup = BeautifulSoup(table_html, 'html.parser')
    trs = soup.find_all('tr')
    occupied: Dict[Tuple[int, int], bool] = {}
    grid: List[List[Tuple[int, str]]] = []

    for ri, tr in enumerate(trs):
        cells = []
        c = 0
        for cell in tr.find_all(['td', 'th']):
            while occupied.get((ri, c)):
                c += 1
            try:
                cs = int(re.sub(r'<[^>]+>', '', str(cell.get('colspan', 1))))
            except (ValueError, TypeError):
                cs = 1
            try:
                rs = int(re.sub(r'<[^>]+>', '', str(cell.get('rowspan', 1))))
            except (ValueError, TypeError):
                rs = 1
            content = cell.decode_contents().strip()
            cells.append((c, content))
            for dr in range(rs):
                for dc in range(cs):
                    occupied[(ri + dr, c + dc)] = True
            c += cs
        grid.append(cells)

    return grid


# ─────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────

class TableFinancialBertTokenizer(FinancialBertTokenizer):
    """
    Extends FinancialBertTokenizer with 2D RoPE position computation.

    HTML tags inside tables are discarded — only cell content is tokenized.
    Each cell is tokenized independently (no cross-cell token merging).
    Returns two extra fields: position_ids_col and position_ids_row.
    """

    def _tokenize_single(
        self, text: str, add_special_tokens: bool = True
    ) -> Dict[str, List]:
        # Split text into (type, content) regions around <table> blocks
        regions: List[Tuple[str, str]] = []
        prev = 0
        for m in _TABLE_RE.finditer(text):
            if m.start() > prev:
                regions.append(('text', text[prev:m.start()]))
            regions.append(('table', m.group(0)))
            prev = m.end()
        if prev < len(text):
            regions.append(('text', text[prev:]))

        all_ids: List[int] = []
        all_is_num: List[int] = []
        all_num_vals: List[List[float]] = []
        all_pos_col: List[float] = []
        all_pos_row: List[float] = []
        seq_pos = 0

        for rtype, content in regions:
            if rtype == 'text':
                seg = super()._tokenize_single(content, add_special_tokens=False)
                n = len(seg["input_ids"])
                all_ids.extend(seg["input_ids"])
                all_is_num.extend(seg["is_number_mask"])
                all_num_vals.extend(seg["number_values"])
                for i in range(n):
                    all_pos_col.append(float(seq_pos + i))
                    all_pos_row.append(float(seq_pos + i))
                seq_pos += n
            else:
                grid = parse_table_grid(content)

                # TABLE_START marker (1D position)
                all_ids.append(TABLE_START_ID)
                all_is_num.append(0)
                all_num_vals.append([0.0, 0.0])
                all_pos_col.append(float(seq_pos))
                all_pos_row.append(float(seq_pos))
                seq_pos += 1

                table_origin = seq_pos
                table_start_idx = len(all_ids)
                total_tokens = 0

                # First pass: tokenize all cells, record max tokens per column
                cell_data: List[Tuple[int, int, Dict, int]] = []
                max_col_tokens: Dict[int, int] = {}
                for ri, row_cells in enumerate(grid):
                    for ci, cell_html in row_cells:
                        seg = super()._tokenize_single(
                            cell_html, add_special_tokens=False
                        )
                        T_c = len(seg["input_ids"])
                        cell_data.append((ri, ci, seg, T_c))
                        max_col_tokens[ci] = max(
                            max_col_tokens.get(ci, 0), T_c
                        )

                # Column-aligned start positions along the row direction
                # Each column slot = max_tokens + 1 (tab delimiter) + D_RC gap
                col_starts: Dict[int, float] = {}
                cumulative = 0.0
                for c in sorted(max_col_tokens.keys()):
                    col_starts[c] = cumulative
                    cumulative += max_col_tokens[c] + 1 + D_RC

                # Second pass: assign 2D positions
                for ri, ci, seg, T_c in cell_data:
                    all_ids.extend(seg["input_ids"])
                    all_is_num.extend(seg["is_number_mask"])
                    all_num_vals.extend(seg["number_values"])

                    base = col_starts[ci]
                    row_off_col = ri * D_COL * COLUMN_DIRECTION_UNIT_VECT[0]
                    row_off_row = ri * D_COL * COLUMN_DIRECTION_UNIT_VECT[1]
                    for t in range(T_c):
                        # Intra-cell: +t on both axes (vanilla RoPE)
                        # Inter-cell: column-aligned base along ROW_DIRECTION
                        # Inter-row: ri * D_COL along COLUMN_DIRECTION
                        all_pos_col.append(
                            table_origin
                            + base * ROW_DIRECTION_UNIT_VECT[0]
                            + row_off_col + t
                        )
                        all_pos_row.append(
                            table_origin
                            + base * ROW_DIRECTION_UNIT_VECT[1]
                            + row_off_row + t
                        )

                    # Tab delimiter after cell content
                    all_ids.append(TAB_ID)
                    all_is_num.append(0)
                    all_num_vals.append([0.0, 0.0])
                    all_pos_col.append(
                        table_origin
                        + base * ROW_DIRECTION_UNIT_VECT[0]
                        + row_off_col + T_c
                    )
                    all_pos_row.append(
                        table_origin
                        + base * ROW_DIRECTION_UNIT_VECT[1]
                        + row_off_row + T_c
                    )
                    total_tokens += T_c + 1  # +1 for tab

                # Center table: place barycenter at midpoint of the
                # position budget (between last-before and first-after)
                if total_tokens > 0:
                    tbl_col = all_pos_col[table_start_idx:table_start_idx + total_tokens]
                    tbl_row = all_pos_row[table_start_idx:table_start_idx + total_tokens]
                    bary_col = sum(tbl_col) / total_tokens
                    bary_row = sum(tbl_row) / total_tokens
                    midpoint = table_origin + (total_tokens - 1) / 2
                    shift_col = midpoint - bary_col
                    shift_row = midpoint - bary_row
                    for i in range(total_tokens):
                        all_pos_col[table_start_idx + i] += shift_col
                        all_pos_row[table_start_idx + i] += shift_row

                seq_pos = table_origin + total_tokens

                # TABLE_END marker (1D position)
                all_ids.append(TABLE_END_ID)
                all_is_num.append(0)
                all_num_vals.append([0.0, 0.0])
                all_pos_col.append(float(seq_pos))
                all_pos_row.append(float(seq_pos))
                seq_pos += 1

        # Add special tokens
        if add_special_tokens:
            cls_id = self.base_tokenizer.cls_token_id
            sep_id = self.base_tokenizer.sep_token_id
            all_ids = [cls_id] + all_ids + [sep_id]
            all_is_num = [0] + all_is_num + [0]
            all_num_vals = [[0.0, 0.0]] + all_num_vals + [[0.0, 0.0]]
            all_pos_col = [0.0] + [p + 1 for p in all_pos_col] + [float(seq_pos + 1)]
            all_pos_row = [0.0] + [p + 1 for p in all_pos_row] + [float(seq_pos + 1)]

        return {
            "input_ids": all_ids,
            "is_number_mask": all_is_num,
            "number_values": all_num_vals,
            "position_ids_col": all_pos_col,
            "position_ids_row": all_pos_row,
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
    Wraps ModernBertRotaryEmbedding to support 2D table positions.

    When 2D positions are set via set_positions(), computes RoPE using
    interleaved axis assignment:
        - even-indexed frequency pairs → column positions (axis 1)
        - odd-indexed frequency pairs  → row positions (axis 2)

    When no 2D positions are set, delegates to the original module.
    The original module stores per-layer-type inv_freq buffers
    (e.g. local_inv_freq, global_inv_freq) as non-persistent buffers,
    so wrapping does not affect state_dict compatibility.
    """

    def __init__(self, original: nn.Module):
        super().__init__()
        self.original = original
        self._pos_col: Optional[torch.Tensor] = None
        self._pos_row: Optional[torch.Tensor] = None

    def set_positions(self, pos_col: torch.Tensor, pos_row: torch.Tensor):
        self._pos_col = pos_col
        self._pos_row = pos_row

    def clear_positions(self):
        self._pos_col = None
        self._pos_row = None

    def forward(self, x, position_ids, layer_type=None):
        if self._pos_col is None:
            return self.original(x, position_ids, layer_type=layer_type)

        # Get the inv_freq for this layer type (local or global)
        inv_freq = getattr(self.original, f"{layer_type}_inv_freq")
        attention_scaling = getattr(
            self.original, f"{layer_type}_attention_scaling"
        )

        # Interleave column and row across frequency bands so both
        # axes see the full frequency spectrum:
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

        cos = emb.cos() * attention_scaling
        sin = emb.sin() * attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


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
