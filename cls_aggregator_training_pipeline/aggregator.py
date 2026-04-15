"""
CLS Token Aggregator.

Inputs: (B, N, D) CLS embeddings from N chunks of a document.
Outputs: (B, N, D) per-position enriched embeddings.

Architecture:
    - Learnable mask token embedding, substituted at masked positions.
    - Pre-norm transformer blocks with RoPE, flash attention (SDPA), SwiGLU FFN.
    - No input projection, no final LayerNorm, no learnable [CLS].

Training (see train.py): two independently-masked views per document, MSE
between their mean-pooled outputs, with SIGReg on the distribution of pooled
embeddings across documents as an anti-collapse regularizer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    return x * cos + _rotate_half(x) * sin


class AggregatorBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, ffn_mult=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.Wqkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.Wo = nn.Linear(hidden_size, hidden_size, bias=False)
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attn_dropout_p = dropout

        ffn_dim = hidden_size * ffn_mult
        self.w_gate = nn.Linear(hidden_size, ffn_dim, bias=False)
        self.w_up = nn.Linear(hidden_size, ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, hidden_size, bias=False)
        self.ffn_drop = nn.Dropout(dropout)

    def forward(self, x, rope_cos, rope_sin, attn_mask=None):
        B, N, D = x.shape
        H, d = self.num_heads, self.head_dim

        h = self.norm1(x)
        qkv = self.Wqkv(h).view(B, N, 3, H, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        dropout_p = self.attn_dropout_p if self.training else 0.0
        h = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
        ).transpose(1, 2).contiguous().view(B, N, D)
        x = x + self.Wo(h)

        h = self.norm2(x)
        x = x + self.ffn_drop(self.w_down(F.silu(self.w_gate(h)) * self.w_up(h)))
        return x


class CLSAggregator(nn.Module):
    """Aggregator over chunk-level CLS embeddings.

    Modules:
        mask_token: learnable embedding, substituted at masked positions.
        layers:     transformer stack, RoPE + flash attention + SwiGLU.
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

        self.mask_token = nn.Parameter(torch.randn(hidden_size))

        self.rope = RotaryPositionEmbedding(hidden_size // num_heads)
        self.layers = nn.ModuleList([
            AggregatorBlock(hidden_size, num_heads, ffn_mult, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, cls_tokens, padding_mask=None, is_masked=None):
        """Run the transformer. Returns (B, N, D) per-position outputs.

        If `is_masked` is given, positions where it is True are replaced by
        the learned mask token before the transformer runs.
        """
        if is_masked is None:
            x = cls_tokens
        else:
            mask_emb = self.mask_token.to(cls_tokens.dtype).view(1, 1, -1).expand_as(cls_tokens)
            x = torch.where(is_masked.unsqueeze(-1), mask_emb, cls_tokens)

        N = x.shape[1]
        rope_cos, rope_sin = self.rope(N, x.device)

        attn_mask = None
        if padding_mask is not None:
            attn_mask = padding_mask.bool()[:, None, None, :]

        for layer in self.layers:
            x = layer(x, rope_cos, rope_sin, attn_mask)
        return x
