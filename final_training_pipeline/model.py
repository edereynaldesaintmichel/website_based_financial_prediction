"""GrowthPredictor: CLSAggregator + classification-regression head."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from cls_aggregator_training_pipeline.aggregator import CLSAggregator


class GrowthPredictor(nn.Module):
    """Predict annual growth rate from chunk-level CLS embeddings.

    Growth rates are discretised into ``n_bins`` bins spanning [min_val, max_val]
    and trained with smoothed cross-entropy (linear interpolation between the
    two nearest bins).
    """

    def __init__(self, aggregator: CLSAggregator = None, hidden_size: int = 768,
                 n_bins: int = 32, min_val: float = -1.0, max_val: float = 1.0):
        super().__init__()
        self.aggregator = aggregator or CLSAggregator(hidden_size=hidden_size)
        self.n_bins = n_bins
        self.min_val = min_val
        self.max_val = max_val

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.GELU(),
            nn.Linear(2 * hidden_size, n_bins),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, cls_tokens, attention_mask=None):
        """
        Args:
            cls_tokens:     (B, N, D)
            attention_mask: (B, N) optional
        Returns:
            dict with ``logits`` (B, n_bins) and ``doc_embedding`` (B, D)
        """
        doc_emb = self.aggregator(cls_tokens, attention_mask=attention_mask)  # (B, D)
        logits = self.head(doc_emb)  # (B, n_bins)
        return {"logits": logits, "doc_embedding": doc_emb}

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(self, logits, targets):
        """Smoothed cross-entropy between logits and continuous growth-rate targets.

        Args:
            logits:  (B, n_bins)
            targets: (B,) — raw annual growth rates
        Returns:
            scalar loss
        """
        norm_pos = (targets.clamp(self.min_val, self.max_val) - self.min_val) / \
                   (self.max_val - self.min_val) * (self.n_bins - 1)

        lower_idx = norm_pos.floor().long()
        upper_idx = norm_pos.ceil().long()

        weight_upper = norm_pos - lower_idx.float()
        weight_lower = 1.0 - weight_upper

        log_probs = F.log_softmax(logits, dim=-1)

        loss = -(weight_lower * log_probs.gather(1, lower_idx.unsqueeze(1)).squeeze(1) +
                 weight_upper * log_probs.gather(1, upper_idx.unsqueeze(1)).squeeze(1))
        return loss.mean()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, cls_tokens, attention_mask=None):
        """Return expected growth rate per sample (weighted sum of bin centres).

        Args:
            cls_tokens:     (B, N, D)
            attention_mask: (B, N) optional
        Returns:
            (B,) expected growth rates
        """
        out = self.forward(cls_tokens, attention_mask=attention_mask)
        probs = F.softmax(out["logits"], dim=-1)  # (B, n_bins)
        centres = torch.linspace(self.min_val, self.max_val, self.n_bins,
                                 device=probs.device)  # (n_bins,)
        return (probs * centres).sum(dim=-1)
