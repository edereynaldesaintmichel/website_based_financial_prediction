# 2D RoPE for Tabular Data — Implementation Spec

## Context

Target model: ModernBERT (head_dim=64, 32 RoPE dimension pairs).
ModernBERT uses two RoPE theta values: local sliding window layers use θ=10,000 (every non-third layer), global attention layers use θ=160,000 (every third layer).
This is a fine-tuning adaptation from a fully trained checkpoint — all design choices prioritize staying close to pretrained RoPE behavior.

## Axis Assignment

The 32 dimension pairs are split across two independent position axes via **interleaving**:
- **Axis 1 (row axis):** even-indexed pairs (0, 2, 4, ..., 30) → 16 pairs
- **Axis 2 (column axis):** odd-indexed pairs (1, 3, 5, ..., 31) → 16 pairs

Interleaving ensures no spectral bias — both axes get a uniform mix of fast-rotating and slow-rotating frequencies.

## Basis Vectors

Three distinct direction vectors are used:

- **Inside cells:** `(1, 1)` — identical to vanilla sequential RoPE. Pretrained token-level attention patterns transfer perfectly.
- **Row direction (cell-to-cell within a row):** `(1, 0.5)` — the inter-cell gap within a row. Axis 1 advances fully, axis 2 advances at half rate.
- **Column direction (row-to-row):** `(0.5, 1)` — the inter-row gap. Axis 1 advances at half rate, axis 2 advances fully.

**Why these vectors:**
- Inside cells uses (1, 1) to preserve vanilla RoPE — a cell containing "John Smith" or "€1,234.56" is just normal text to the model.
- Row and column directions are symmetric mirrors of each other: (1, 0.5) and (0.5, 1). This produces nearly identical RoPE score drops for horizontal and vertical cell crossings (balance diff = 0.01 on local layers).
- All position deltas are strictly positive on both axes. This keeps us in the pretrained regime — the learned Q/K rotational offsets (φ_i) never encounter negative-delta regions they haven't been trained on.
- The (0.5, 1) column direction avoids the axis-1 floor problem that (0, 1) would cause: with (0, 1), same-column tokens share axis 1 perfectly (score floor of 16/32), making tables structurally too flat in attention regardless of spacing.

## Spacing Parameters

- **D_rc = 6** — inter-cell gap within a row, measured along the (1, 0.5) direction. The position delta from the last token of one cell to the first token of the next cell in the same row is `(D_rc + 1, D_rc * 0.5 + 1) = (7, 4)` (the +1 on each axis comes from the natural one-token step). Cross-cell score: **23.70**/32 on local layers.
- **D_col = 7** — inter-row spacing, measured along the (0.5, 1) direction. The position delta from a token to the same-position token one row down is `(D_col * 0.5, D_col * 1.0) = (3.5, 7.0)`. Cross-row score: **23.69**/32 on local layers.

**Balance:** crossing a cell boundary within a row produces essentially the same RoPE score drop as crossing a row boundary within a column (diff = 0.01 on local, 0.17 on global). Both transitions mean "moving to a different cell."

**Score hierarchy (local layers, θ=10k):**

| Relationship | Δp1 | Δp2 | Score | 1D equiv |
|---|---|---|---|---|
| Same cell, adjacent tokens | 1 | 1 | 30.92 | ~1 tok |
| Same cell, opposite ends (T=7) | 6 | 6 | 23.56 | ~6 tok |
| Adjacent cell, same row | 7 | 4 | 23.70 | ~4 tok |
| Same column, 1 row down | 3.5 | 7 | 23.69 | ~4 tok |
| Same column, 2 rows down | 7 | 14 | 21.85 | ~9 tok |
| Diagonal (1 right, 1 down) | 16.5 | 17 | 19.41 | ~20 tok |
| 2 cells right, same row | 26 | 20 | 16.92 | ~30 tok |

## Position Computation

### Outside tables (normal text)
Standard RoPE: both axes increment by 1 per token.
```
p1 = sequential_position
p2 = sequential_position
```

### Inside a table

Token at **(row r, col c, intra-cell token index t)** where t ∈ [0, T_c - 1] and T_c is the token count of cell (r, c):

```
within_row_p1 = sum(T_0 ... T_{c-1}) + c * D_rc + t
within_row_p2 = sum(T_0 ... T_{c-1}) + c * D_rc * 0.5 + t

p1 = within_row_p1 + r * D_col * 0.5
p2 = within_row_p2 + r * D_col * 1.0
```

Where `sum(T_0 ... T_{c-1})` is the cumulative token count of all cells to the left in the same row (0 for the first cell).

For fixed-size cells (T tokens each), this simplifies to:
```
p1 = c * (T + D_rc) + t + r * D_col * 0.5
p2 = c * (T + D_rc * 0.5) + t + r * D_col * 1.0
```

### Within a cell
Tokens advance at `(+1, +1)` per token — **vanilla RoPE**. The pretrained model's token-level attention patterns transfer perfectly inside cells.

### Table entry and exit
- The table's internal positions are computed relative to an origin at the current sequential position in the document.
- After the table, text resumes at `position_before_table + total_token_count_of_table`.
- No explicit margin is needed: table/tr tag overhead tokens provide natural margin. The 2D layout is always more compact than the 1D token count for all table sizes (even a 2×2 table with T=7 has a surplus of +5.0 on both axes).

## Budget Analysis

For T=7 tokens/cell, D_rc=6, D_col=7:

| Table | R×C | Tokens | Max p1 | Max p2 | Surplus p1 | Surplus p2 |
|---|---|---|---|---|---|---|
| Tiny | 2×2 | 28 | 22.5 | 23.0 | +5.5 | +5.0 |
| Small | 3×3 | 63 | 39.0 | 40.0 | +24.0 | +23.0 |
| Medium | 5×5 | 175 | 72.0 | 74.0 | +103.0 | +101.0 |
| Large | 10×10 | 700 | 154.5 | 159.0 | +545.5 | +541.0 |
| Huge | 20×10 | 1400 | 189.5 | 229.0 | +1210.5 | +1171.0 |
| Max | 30×15 | 3150 | 289.5 | 349.0 | +2860.5 | +2801.0 |

All tables have positive surplus on both axes. Large tables are extremely budget-efficient due to 2D compactness (a 20×10 table uses only ~229 positions on its widest axis for 1400 tokens — 84% savings). The two axes act as independent budgets: a wide table uses more axis 1 budget, a tall table uses more axis 2 budget.

## Neighbor Density

With these parameters, a token inside a table has roughly **3–5× more "close" RoPE neighbors** than a token in 1D text of the same length (measured as tokens within 8 raw score points, which corresponds to ~1 unit in softmax after the √d_k scaling). This is a structural consequence of 2D layout and is appropriate: table cells genuinely have more spatial neighbors than sequential text.

## Key Design Invariants

1. **Within-cell = vanilla RoPE.** Intra-cell tokens advance (1, 1). No modification needed.
2. **All position deltas are positive.** No token ever has a lower position than a preceding token on either axis. This keeps us in the pretrained regime.
3. **Cell boundaries are balanced.** Horizontal and vertical cell crossings produce nearly identical RoPE score drops (23.70 vs 23.69 on local layers).
4. **Interleaved axis assignment.** Even pairs → axis 1, odd pairs → axis 2. No spectral bias.
5. **Symmetric direction vectors.** Row direction (1, 0.5) and column direction (0.5, 1) are mirrors, reflecting the symmetric role of horizontal and vertical cell relationships.