"""
TableFinancialModernBert: FinancialModernBert with learned 2D positional
embeddings for HTML tables.

Each attention layer has two small embedding tables — one for row position,
one for column position — whose outputs are added to hidden states before the
QKV projection.  Standard 1D RoPE still runs on all tokens for sequential
position; the learned embeddings provide *additional* structural row/column
information for table tokens only.

Tables with more rows (or columns) than the embedding table size are handled
via proportional remapping + linear interpolation between the two nearest
embeddings (similar to ViT position embedding interpolation at new resolutions).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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

# Default limits for position embedding tables
MAX_ROW_POSITIONS = 40
MAX_COL_POSITIONS = 8


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
# Position embedding helpers
# ─────────────────────────────────────────────────────────────

def _interpolate_embedding(
    emb: nn.Embedding,
    indices: torch.Tensor,
    num_positions: torch.Tensor,
    max_pos: int,
) -> torch.Tensor:
    """Look up or interpolate positional embeddings.

    For tables with <= max_pos positions along an axis, uses direct lookup.
    For tables with > max_pos, remaps ALL indices proportionally to
    [0, max_pos-1] and linearly interpolates between the two nearest
    embeddings.
    """
    needs_interp = num_positions > max_pos

    # Proportional remapping for large tables
    denom = (num_positions.float() - 1).clamp(min=1)
    remapped = indices.float() * (max_pos - 1) / denom

    # Direct indices for small tables, remapped for large
    effective = torch.where(needs_interp, remapped, indices.float())

    idx_lo = effective.floor().long().clamp(0, max_pos - 1)
    idx_hi = effective.ceil().long().clamp(0, max_pos - 1)
    weight_hi = (effective - idx_lo.float()).unsqueeze(-1)

    return (1.0 - weight_hi) * emb(idx_lo) + weight_hi * emb(idx_hi)


# ─────────────────────────────────────────────────────────────
# Attention wrapper
# ─────────────────────────────────────────────────────────────

class TablePosAttentionWrapper(nn.Module):
    """Wraps ModernBertAttention to inject learned table position embeddings.

    Uses explicit linear separation:
        QKV = Wqkv(hidden) + pos @ Wqkv.weight
    The position contribution is computed via a separate F.linear call
    (no bias — already present in the hidden path) and added to the Wqkv
    output through a forward hook.  When use_value_pos is False, the V
    slice of the position contribution is zeroed before addition.
    """

    def __init__(
        self,
        original_attn: nn.Module,
        hidden_size: int,
        max_row_pos: int,
        max_col_pos: int,
        use_value_pos: bool,
        num_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.original = original_attn
        self.row_emb = nn.Embedding(max_row_pos, hidden_size)
        self.col_emb = nn.Embedding(max_col_pos, hidden_size)
        self.max_row_pos = max_row_pos
        self.max_col_pos = max_col_pos
        self.use_value_pos = use_value_pos
        self.num_heads = num_heads
        self.head_dim = head_dim

        nn.init.zeros_(self.row_emb.weight)
        nn.init.zeros_(self.col_emb.weight)

        self._table_pos: Optional[Tuple] = None

    def set_table_pos(self, row_idx, col_idx, table_mask, num_rows, num_cols):
        self._table_pos = (row_idx, col_idx, table_mask, num_rows, num_cols)

    def clear_table_pos(self):
        self._table_pos = None

    def _compute_pos_emb(self, dtype: torch.dtype) -> torch.Tensor:
        row_idx, col_idx, table_mask, num_rows, num_cols = self._table_pos
        row_e = _interpolate_embedding(
            self.row_emb, row_idx, num_rows, self.max_row_pos,
        )
        col_e = _interpolate_embedding(
            self.col_emb, col_idx, num_cols, self.max_col_pos,
        )
        return ((row_e + col_e) * table_mask.unsqueeze(-1)).to(dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self._table_pos is None:
            return self.original(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                **kwargs,
            )

        pos_emb = self._compute_pos_emb(hidden_states.dtype)

        # pos @ Wqkv.weight (no bias — already in the hidden path)
        pos_qkv = F.linear(pos_emb, self.original.Wqkv.weight)

        if not self.use_value_pos:
            B, S = pos_qkv.shape[:2]
            qkv_mask = pos_qkv.new_ones(3)
            qkv_mask[2] = 0.0  # zero V
            pos_qkv = (
                pos_qkv.view(B, S, 3, self.num_heads, self.head_dim)
                * qkv_mask[None, None, :, None, None]
            ).view(B, S, -1)

        # Hook on Wqkv to add position contribution to its output
        handle = self.original.Wqkv.register_forward_hook(
            lambda _mod, _inp, out: out + pos_qkv
        )
        try:
            result = self.original(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                **kwargs,
            )
        finally:
            handle.remove()

        return result


# ─────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────

class TableFinancialBertTokenizer(FinancialBertTokenizer):
    """
    Extends FinancialBertTokenizer with integer table row/column indices.

    HTML tags inside tables are discarded — only cell content is tokenized.
    Each cell is tokenized independently (no cross-cell token merging).
    Returns extra fields: table_row_index, table_col_index, table_mask,
    table_num_rows, table_num_cols.
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
        all_num_vals: List = []
        all_row_idx: List[int] = []
        all_col_idx: List[int] = []
        all_table_mask: List[int] = []
        all_num_rows: List[int] = []
        all_num_cols: List[int] = []

        for rtype, content in regions:
            if rtype == 'text':
                seg = super()._tokenize_single(content, add_special_tokens=False)
                n = len(seg["input_ids"])
                all_ids.extend(seg["input_ids"])
                all_is_num.extend(seg["is_number_mask"])
                all_num_vals.extend(seg["number_values"])
                all_row_idx.extend([0] * n)
                all_col_idx.extend([0] * n)
                all_table_mask.extend([0] * n)
                all_num_rows.extend([0] * n)
                all_num_cols.extend([0] * n)
            else:
                grid = parse_table_grid(content)
                n_rows = len(grid)
                n_cols = max(
                    (ci for row in grid for ci, _ in row), default=-1
                ) + 1

                # TABLE_START marker (not a table cell — no position)
                all_ids.append(TABLE_START_ID)
                all_is_num.append(0)
                all_num_vals.append(0.0)
                all_row_idx.append(0)
                all_col_idx.append(0)
                all_table_mask.append(0)
                all_num_rows.append(0)
                all_num_cols.append(0)

                for ri, row_cells in enumerate(grid):
                    for ci, cell_html in row_cells:
                        seg = super()._tokenize_single(
                            cell_html, add_special_tokens=False
                        )
                        T_c = len(seg["input_ids"])
                        all_ids.extend(seg["input_ids"])
                        all_is_num.extend(seg["is_number_mask"])
                        all_num_vals.extend(seg["number_values"])
                        all_row_idx.extend([ri] * T_c)
                        all_col_idx.extend([ci] * T_c)
                        all_table_mask.extend([1] * T_c)
                        all_num_rows.extend([n_rows] * T_c)
                        all_num_cols.extend([n_cols] * T_c)

                        # Tab delimiter (same row/col as preceding cell)
                        all_ids.append(TAB_ID)
                        all_is_num.append(0)
                        all_num_vals.append(0.0)
                        all_row_idx.append(ri)
                        all_col_idx.append(ci)
                        all_table_mask.append(1)
                        all_num_rows.append(n_rows)
                        all_num_cols.append(n_cols)

                # TABLE_END marker
                all_ids.append(TABLE_END_ID)
                all_is_num.append(0)
                all_num_vals.append(0.0)
                all_row_idx.append(0)
                all_col_idx.append(0)
                all_table_mask.append(0)
                all_num_rows.append(0)
                all_num_cols.append(0)

        # Add special tokens
        if add_special_tokens:
            cls_id = self.base_tokenizer.cls_token_id
            sep_id = self.base_tokenizer.sep_token_id
            all_ids = [cls_id] + all_ids + [sep_id]
            all_is_num = [0] + all_is_num + [0]
            all_num_vals = [0.0] + all_num_vals + [0.0]
            all_row_idx = [0] + all_row_idx + [0]
            all_col_idx = [0] + all_col_idx + [0]
            all_table_mask = [0] + all_table_mask + [0]
            all_num_rows = [0] + all_num_rows + [0]
            all_num_cols = [0] + all_num_cols + [0]

        return {
            "input_ids": all_ids,
            "is_number_mask": all_is_num,
            "number_values": all_num_vals,
            "table_row_index": all_row_idx,
            "table_col_index": all_col_idx,
            "table_mask": all_table_mask,
            "table_num_rows": all_num_rows,
            "table_num_cols": all_num_cols,
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
            "table_row_index", "table_col_index", "table_mask",
            "table_num_rows", "table_num_cols",
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
                batched["number_values"][i] += [0.0] * pad_len
                batched["table_row_index"][i] += [0] * pad_len
                batched["table_col_index"][i] += [0] * pad_len
                batched["table_mask"][i] += [0] * pad_len
                batched["table_num_rows"][i] += [0] * pad_len
                batched["table_num_cols"][i] += [0] * pad_len

            batch_attention_mask.append(attn_mask)

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(
                    batched["input_ids"], dtype=torch.long,
                ),
                "attention_mask": torch.tensor(
                    batch_attention_mask, dtype=torch.long,
                ),
                "is_number_mask": torch.tensor(
                    batched["is_number_mask"], dtype=torch.float,
                ),
                "number_values": torch.tensor(
                    batched["number_values"], dtype=torch.float,
                ),
                "table_row_index": torch.tensor(
                    batched["table_row_index"], dtype=torch.long,
                ),
                "table_col_index": torch.tensor(
                    batched["table_col_index"], dtype=torch.long,
                ),
                "table_mask": torch.tensor(
                    batched["table_mask"], dtype=torch.float,
                ),
                "table_num_rows": torch.tensor(
                    batched["table_num_rows"], dtype=torch.long,
                ),
                "table_num_cols": torch.tensor(
                    batched["table_num_cols"], dtype=torch.long,
                ),
            }

        return {
            "input_ids": batched["input_ids"],
            "attention_mask": batch_attention_mask,
            "is_number_mask": batched["is_number_mask"],
            "number_values": batched["number_values"],
            "table_row_index": batched["table_row_index"],
            "table_col_index": batched["table_col_index"],
            "table_mask": batched["table_mask"],
            "table_num_rows": batched["table_num_rows"],
            "table_num_cols": batched["table_num_cols"],
        }


# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────

class TableFinancialModernBert(FinancialModernBert):
    """FinancialModernBert with learned 2D positional embeddings for tables."""

    config_class = FinancialModernBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.max_row_pos = getattr(config, 'max_row_positions', MAX_ROW_POSITIONS)
        self.max_col_pos = getattr(config, 'max_col_positions', MAX_COL_POSITIONS)
        self.use_value_pos = getattr(config, 'use_value_pos', True)
        self._wrap_attention_layers()

    def _wrap_attention_layers(self):
        """Wrap each attention module with TablePosAttentionWrapper."""
        num_heads = self.config.num_attention_heads
        head_dim = self.config.hidden_size // num_heads

        for layer in self.modernbert.layers:
            layer.attn = TablePosAttentionWrapper(
                original_attn=layer.attn,
                hidden_size=self.config.hidden_size,
                max_row_pos=self.max_row_pos,
                max_col_pos=self.max_col_pos,
                use_value_pos=self.use_value_pos,
                num_heads=num_heads,
                head_dim=head_dim,
            )

    def _get_attention_wrappers(self) -> List[TablePosAttentionWrapper]:
        return [
            layer.attn for layer in self.modernbert.layers
            if isinstance(layer.attn, TablePosAttentionWrapper)
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        number_values: torch.Tensor,
        is_number_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        table_row_index: Optional[torch.Tensor] = None,
        table_col_index: Optional[torch.Tensor] = None,
        table_mask: Optional[torch.Tensor] = None,
        table_num_rows: Optional[torch.Tensor] = None,
        table_num_cols: Optional[torch.Tensor] = None,
        labels_text: Optional[torch.Tensor] = None,
        labels_magnitude: Optional[torch.Tensor] = None,
    ):
        # Set table positions on all attention wrappers
        if table_mask is not None and table_mask.any():
            for w in self._get_attention_wrappers():
                w.set_table_pos(
                    table_row_index, table_col_index, table_mask,
                    table_num_rows, table_num_cols,
                )

        try:
            return super().forward(
                input_ids=input_ids,
                number_values=number_values,
                is_number_mask=is_number_mask,
                attention_mask=attention_mask,
                labels_text=labels_text,
                labels_magnitude=labels_magnitude,
            )
        finally:
            for w in self._get_attention_wrappers():
                w.clear_table_pos()


def build_table_model(
    model_id: str = "answerdotai/ModernBERT-base",
    num_magnitude_bins: int = 128,
    max_row_positions: int = MAX_ROW_POSITIONS,
    max_col_positions: int = MAX_COL_POSITIONS,
    use_value_pos: bool = True,
    **kwargs,
) -> TableFinancialModernBert:
    """Build a TableFinancialModernBert from a pretrained ModernBERT."""
    donor = ModernBertForMaskedLM.from_pretrained(model_id)

    config = FinancialModernBertConfig.from_pretrained(model_id)
    config.num_magnitude_bins = num_magnitude_bins
    config.max_row_positions = max_row_positions
    config.max_col_positions = max_col_positions
    config.use_value_pos = use_value_pos

    model = TableFinancialModernBert(config)

    # Remap donor keys: attention weights live under .attn.original. now
    donor_sd = donor.model.state_dict()
    remapped = {}
    for k, v in donor_sd.items():
        if '.attn.' in k:
            k = k.replace('.attn.', '.attn.original.', 1)
        remapped[k] = v

    # strict=False: new row_emb/col_emb weights have no donor counterpart
    model.modernbert.load_state_dict(remapped, strict=False)
    model.lm_head.dense.load_state_dict(donor.head.dense.state_dict())
    model.lm_head.norm.load_state_dict(donor.head.norm.state_dict())

    embedding_layer = model._get_embedding_layer()
    model.lm_head.decoder.weight = embedding_layer.weight
    model.lm_head.decoder.bias.data = donor.decoder.bias.data.clone()

    return model
