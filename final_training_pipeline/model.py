"""GrowthPredictor: CLSAggregator (JEPA-trained, mean-pooled) + regression head."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
import torch.nn.functional as F

from cls_aggregator_training_pipeline.aggregator import CLSAggregator


class GrowthPredictor(nn.Module):
    """Predict annual growth rate from chunk-level CLS embeddings.

    The aggregator was trained with a JEPA-adjacent latent-MLM objective and
    returns per-position outputs (B, N, D). We mean-pool over valid positions
    to form a document embedding, then pass through a regression head.
    """

    def __init__(self, aggregator: CLSAggregator = None, hidden_size: int = 768):
        super().__init__()
        self.aggregator = aggregator or CLSAggregator(hidden_size=hidden_size)

        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, cls_tokens, attention_mask=None):
        """
        Args:
            cls_tokens:     (B, N, D)
            attention_mask: (B, N) 1=valid, 0=pad (optional)
        Returns:
            dict with ``prediction`` (B,) and ``doc_embedding`` (B, D)
        """
        # Inference mode (is_masked=None) -> per-position outputs (B, N, D)
        per_pos = self.aggregator(cls_tokens, padding_mask=attention_mask)

        if attention_mask is not None:
            m = attention_mask.to(per_pos.dtype).unsqueeze(-1)  # (B, N, 1)
            doc_emb = (per_pos * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            doc_emb = per_pos.mean(dim=1)

        prediction = self.head(doc_emb).squeeze(-1)
        return {"prediction": prediction, "doc_embedding": doc_emb}

    def compute_loss(self, prediction, targets):
        return F.mse_loss(prediction, targets)
