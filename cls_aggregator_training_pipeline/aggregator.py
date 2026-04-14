"""
CLS Token Aggregator for latent MLM.

Inputs: (B, N, D) CLS embeddings from N chunks of a document.
Outputs: (B, N, D) per-position enriched embeddings.

Architecture:
    - Shared linear projection W (init as identity, bias-free). Applied to CLS
      tokens to produce the latent space in which prediction and regularization
      happen. At inference, mean-pool the aggregator outputs.
    - Learnable mask token embedding (in the W-projected space).
    - Pre-norm transformer blocks with RoPE, flash attention (SDPA), SwiGLU FFN.
    - No learnable [CLS] — we predict at every position.

Training (not implemented here; see train.py):
    target = sg(W(cls))                                  # stop-grad, frozen teacher
    x_in[i]  = mask_token     if masked[i]
             = target[i]      otherwise
    pred     = transformer(x_in)
    loss     = MSE(pred[masked], target[masked]) + λ · SIGReg(pred)
    W is trained only through the aggregator input path for unmasked positions.
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
    """Latent-MLM aggregator over chunk-level CLS embeddings.

    Modules:
        W:          shared linear projection cls → latent, bias=False, init=I.
        mask_token: learnable embedding in latent space, replaces masked targets.
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

        self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            self.W.weight.copy_(torch.eye(hidden_size))

        self.mask_token = nn.Parameter(torch.randn(hidden_size) * 0.02)

        self.rope = RotaryPositionEmbedding(hidden_size // num_heads)
        self.layers = nn.ModuleList([
            AggregatorBlock(hidden_size, num_heads, ffn_mult, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_size)

    def project(self, cls_tokens):
        """Apply the shared W projection: (B, N, D) → (B, N, D)."""
        return self.W(cls_tokens)

    def forward_latent(self, latent, padding_mask=None):
        """Run the transformer on already-projected latent sequences.

        Args:
            latent:       (B, N, D) latent-space tokens (mask tokens substituted).
            padding_mask: (B, N) 1=valid, 0=pad. None = no padding.
        Returns:
            (B, N, D) per-position outputs.
        """
        N = latent.shape[1]
        rope_cos, rope_sin = self.rope(N, latent.device)

        attn_mask = None
        if padding_mask is not None:
            attn_mask = padding_mask.bool()[:, None, None, :]

        x = latent
        for layer in self.layers:
            x = layer(x, rope_cos, rope_sin, attn_mask)
        return self.final_norm(x)

    def forward(self, cls_tokens, padding_mask=None, is_masked=None):
        """Run the aggregator on CLS embeddings.

        Training mode (`is_masked` given):
            target = sg(W(cls))
            x_in   = mask_token where is_masked else W(cls) (with grad on W)
            pred   = transformer(x_in)
            returns (pred, target)

        Inference mode (`is_masked` is None):
            returns pred of shape (B, N, D). Caller mean-pools over valid
            positions to obtain a document embedding.
        """
        projected = self.project(cls_tokens)
        if is_masked is None:
            return self.forward_latent(projected, padding_mask)

        target = projected.detach()
        mask_emb = self.mask_token.to(projected.dtype).view(1, 1, -1).expand_as(projected)
        x_in = torch.where(is_masked.unsqueeze(-1), mask_emb, projected)
        pred = self.forward_latent(x_in, padding_mask)
        return pred, target
