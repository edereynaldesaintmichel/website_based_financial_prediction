"""
CLS Token Aggregator: lightweight transformer over chunk-level CLS embeddings.

Takes N CLS tokens (one per chunk of a long document) and produces N enriched,
context-aware CLS tokens via self-attention with ALiBi positional encoding
(learnable decay rate per head).

Architecture (easily configurable):
    - 6 layers, 768 hidden dim, 16 attention heads
    - Pre-norm transformer blocks (LayerNorm → Attention → residual → LayerNorm → FFN → residual)
    - ALiBi with learnable log-slope per head (most heads learn ~0 decay)
    - SwiGLU FFN (4x expansion)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableALiBi(nn.Module):
    """ALiBi positional bias with a learnable log-slope per head.

    Each head has a scalar log_slope (initialized to the standard ALiBi
    geometric schedule). The bias for a (query, key) pair at positions
    (i, j) is:  -exp(log_slope) * |i - j|.
    """

    def __init__(self, num_heads):
        super().__init__()
        # Standard ALiBi initialization: geometric slopes
        ratio = 2 ** (-8.0 / num_heads)
        slopes = torch.tensor([ratio ** (i + 1) for i in range(num_heads)])
        self.log_slopes = nn.Parameter(slopes.log())  # (H,)

    def forward(self, seq_len, device):
        """Returns ALiBi bias of shape (1, H, S, S)."""
        positions = torch.arange(seq_len, device=device)
        # |i - j| distance matrix
        dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs().float()  # (S, S)
        slopes = self.log_slopes.exp()  # (H,)
        # (H, S, S) → (1, H, S, S)
        return -(slopes[:, None, None] * dist[None, :, :]).unsqueeze(0)


class AggregatorBlock(nn.Module):
    """Pre-norm transformer block with SwiGLU FFN."""

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

    def forward(self, x, alibi_bias):
        """
        Args:
            x: (B, N, D) sequence of CLS tokens
            alibi_bias: (1, H, N, N) ALiBi attention bias
        """
        B, N, D = x.shape
        H, d = self.num_heads, self.head_dim

        # Self-attention (uses memory-efficient / flash attention via SDPA)
        h = self.norm1(x)
        qkv = self.Wqkv(h).view(B, N, 3, H, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, N, d)

        dropout_p = self.attn_dropout_p if self.training else 0.0
        h = F.scaled_dot_product_attention(
            q, k, v, attn_mask=alibi_bias, dropout_p=dropout_p,
        ).transpose(1, 2).contiguous().view(B, N, D)
        x = x + self.Wo(h)

        # SwiGLU FFN
        h = self.norm2(x)
        x = x + self.ffn_drop(self.w_down(F.silu(self.w_gate(h)) * self.w_up(h)))

        return x


class CLSAggregator(nn.Module):
    """Transformer aggregator over chunk-level CLS embeddings.

    Prepends a learnable CLS token to the N chunk embeddings, runs
    self-attention with ALiBi, and returns the CLS output as a single
    document-level representation.

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

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.alibi = LearnableALiBi(num_heads)
        self.layers = nn.ModuleList([
            AggregatorBlock(hidden_size, num_heads, ffn_mult, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(self, cls_tokens):
        """
        Args:
            cls_tokens: (B, N, D) CLS embeddings from N chunks
        Returns:
            (B, D) single document embedding
        """
        B, N, D = cls_tokens.shape

        # Prepend learnable CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls, cls_tokens], dim=1)  # (B, 1+N, D)

        alibi_bias = self.alibi(1 + N, cls_tokens.device)

        for layer in self.layers:
            x = layer(x, alibi_bias)

        return self.final_norm(x[:, 0, :])  # (B, D)
