"""
CLS Token Aggregator: lightweight transformer over chunk-level CLS embeddings.

Takes N CLS tokens (one per chunk of a long document) and produces N enriched,
context-aware CLS tokens via self-attention with RoPE positional encoding
and flash attention.

Architecture (easily configurable):
    - 6 layers, 768 hidden dim, 16 attention heads
    - Pre-norm transformer blocks (LayerNorm → Attention → residual → LayerNorm → FFN → residual)
    - Rotary Position Embeddings (RoPE)
    - Flash Attention via SDPA
    - SwiGLU FFN (4x expansion)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionEmbedding(nn.Module):
    """RoPE: encodes position by rotating (q, k) vectors in 2D sub-spaces."""

    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        """Returns cos/sin of shape (S, D) for apply_rope."""
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (S, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (S, head_dim)
        return emb.cos(), emb.sin()


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    """Apply RoPE to x: (B, H, S, d)."""
    return x * cos + _rotate_half(x) * sin


class AggregatorBlock(nn.Module):
    """Pre-norm transformer block with SwiGLU FFN and flash attention."""

    def __init__(self, hidden_size, num_heads, ffn_mult=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        # Multi-head self-attention (no bias, like ModernBERT)
        self.Wqkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.Wo = nn.Linear(hidden_size, hidden_size, bias=False)
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attn_dropout_p = dropout

        # SwiGLU FFN
        ffn_dim = hidden_size * ffn_mult
        self.w_gate = nn.Linear(hidden_size, ffn_dim, bias=False)
        self.w_up = nn.Linear(hidden_size, ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, hidden_size, bias=False)
        self.ffn_drop = nn.Dropout(dropout)

    def forward(self, x, rope_cos, rope_sin, attn_mask=None):
        """
        Args:
            x: (B, N, D) sequence of CLS tokens
            rope_cos, rope_sin: (S, d) precomputed RoPE frequencies
            attn_mask: (B, 1, 1, N) boolean mask (True=valid) for padding, or None
        """
        B, N, D = x.shape
        H, d = self.num_heads, self.head_dim

        # Self-attention
        h = self.norm1(x)
        qkv = self.Wqkv(h).view(B, N, 3, H, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, N, d)

        # Apply RoPE to q and k
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        dropout_p = self.attn_dropout_p if self.training else 0.0
        h = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
            enable_gqa=False,
        ).transpose(1, 2).contiguous().view(B, N, D)
        x = x + self.Wo(h)

        # SwiGLU FFN
        h = self.norm2(x)
        x = x + self.ffn_drop(self.w_down(F.silu(self.w_gate(h)) * self.w_up(h)))

        return x


class CLSAggregator(nn.Module):
    """Transformer aggregator over chunk-level CLS embeddings.

    Prepends a learnable CLS token to the N chunk embeddings, runs
    bidirectional self-attention with RoPE and flash attention, and
    returns the CLS output as a single document-level representation.

    Input:  (B, N, D) — N CLS tokens from N chunks
    Output: (B, D)    — single document embedding
    """

    def __init__(
        self,
        hidden_size=768,
        num_heads=16,
        num_layers=6,
        ffn_mult=4,
        dropout=0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.cls_token = nn.Parameter(torch.randn(1, hidden_size) * 0.02)
        self.rope = RotaryPositionEmbedding(hidden_size // num_heads)
        self.layers = nn.ModuleList([
            AggregatorBlock(hidden_size, num_heads, ffn_mult, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(self, cls_tokens, attention_mask=None):
        """
        Args:
            cls_tokens: (B, N, D) CLS embeddings from N chunks
            attention_mask: (B, N) optional padding mask (1=valid, 0=pad)
        Returns:
            (B, D) single document embedding
        """
        B, N, D = cls_tokens.shape

        # Prepend learnable CLS token
        cls = self.cls_token.unsqueeze(0).expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls, cls_tokens], dim=1)  # (B, 1+N, D)

        # RoPE frequencies for sequence length 1+N
        rope_cos, rope_sin = self.rope(1 + N, cls_tokens.device)

        # Build boolean padding mask for flash attention if needed
        attn_mask = None
        if attention_mask is not None:
            # Prepend 1 for the learnable CLS token (always valid)
            mask = torch.cat([
                attention_mask.new_ones(B, 1), attention_mask
            ], dim=1).bool()  # (B, 1+N)
            # Key-side mask: (B, 1, 1, 1+N) — broadcastable over heads and queries
            attn_mask = mask[:, None, None, :]

        for layer in self.layers:
            x = layer(x, rope_cos, rope_sin, attn_mask)

        return self.final_norm(x[:, 0, :])  # (B, D)
