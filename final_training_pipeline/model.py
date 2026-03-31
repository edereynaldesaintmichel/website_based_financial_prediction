"""GrowthPredictor: CLSAggregator + regression head."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from cls_aggregator_training_pipeline.aggregator import CLSAggregator


class GrowthPredictor(nn.Module):
    """Predict annual growth rate from chunk-level CLS embeddings (regression)."""

    def __init__(self, aggregator: CLSAggregator = None, hidden_size: int = 768):
        super().__init__()
        self.aggregator = aggregator or CLSAggregator(hidden_size=hidden_size, dropout=0.4)

        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 4*hidden_size),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(4*hidden_size, 1),
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
            dict with ``prediction`` (B,) and ``doc_embedding`` (B, D)
        """
        doc_emb = self.aggregator(cls_tokens, attention_mask=attention_mask)  # (B, D)
        prediction = self.head(doc_emb).squeeze(-1)  # (B,)
        return {"prediction": prediction, "doc_embedding": doc_emb}

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(self, prediction, targets):
        """MSE loss between predicted and actual growth rates.

        Args:
            prediction: (B,) — predicted growth rates
            targets:    (B,) — actual growth rates
        Returns:
            scalar loss
        """
        return F.mse_loss(prediction, targets)
