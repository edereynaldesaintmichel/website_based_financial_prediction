"""
MLM Decoder with expanded-memory cross-attention for encoder CLS injection.

Architecture per decoder layer:
    ExpandedMemoryCrossAttention(h, CLS) → SelfAttn + FFN

- Self-attention + FFN: initialized from pretrained ModernBERT
- CLS conditioning: multi-head cross-attention to expanded CLS memory.
  CLS (B, D) is projected into N memory slots, then standard multi-head
  cross-attention lets each decoder position attend to different slots
  based on content. Output projection zero-initialized so conditioning
  starts as a no-op.
- Number prediction: same magnitude-bin loss as the encoder's NumberHead

The decoder is bidirectional (MLM, not autoregressive). During training, ~50% of
tokens are masked and the decoder predicts them using CLS + unmasked context.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ModernBertModel, ModernBertForMaskedLM

from transformers.masking_utils import (
    create_bidirectional_mask,
    create_bidirectional_sliding_window_mask,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from financial_bert import (
    FinancialModernBert,
    FinancialModernBertConfig,
    NumberEmbedder,
    NumberHead,
    PredictionHead,
)

NUM_MEMORY_SLOTS = 16

class ExpandedMemoryCrossAttention(nn.Module):
    """Multi-head cross-attention to expanded CLS memory.

    CLS (B, D) is projected into N memory slots (B, N, D), then standard
    multi-head cross-attention lets each decoder position attend to different
    slots based on content. This gives position-dependent retrieval from CLS,
    unlike single-vector approaches where every position gets the same info.
    """

    def __init__(self, config, num_slots=NUM_MEMORY_SLOTS):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_slots = num_slots
        D = config.hidden_size

        # Expand CLS into N memory slots
        self.W_expand = nn.Linear(D, num_slots * D, bias=False)

        # Standard cross-attention projections
        self.Wq = nn.Linear(D, D, bias=False)
        self.Wk = nn.Linear(D, D, bias=False)
        self.Wv = nn.Linear(D, D, bias=False)
        self.Wo = nn.Linear(D, D, bias=False)

        # Zero-init output so cross-attention starts as a no-op
        nn.init.zeros_(self.Wo.weight)

    def forward(self, hidden_states, cls_hidden):
        """
        Args:
            hidden_states: (B, S, D) decoder hidden states
            cls_hidden: (B, D) encoder CLS representation
        Returns:
            (B, S, D) cross-attention output
        """
        B, S, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        N = self.num_slots

        # Expand CLS into memory slots
        memory = self.W_expand(cls_hidden).view(B, N, D)  # (B, N, D)

        q = self.Wq(hidden_states).view(B, S, H, d).transpose(1, 2)  # (B, H, S, d)
        k = self.Wk(memory).view(B, N, H, d).transpose(1, 2)          # (B, H, N, d)
        v = self.Wv(memory).view(B, N, H, d).transpose(1, 2)          # (B, H, N, d)

        attn_out = F.scaled_dot_product_attention(q, k, v)  # (B, H, S, d)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)

        return self.Wo(attn_out)
    


class CLSBottleneckDecoder(nn.Module):
    """Bidirectional decoder with expanded-memory CLS conditioning.

    Uses a full ModernBertModel as backbone (reusing its layers, embeddings,
    rotary embeddings, and attention mask logic). CLS conditioning is injected
    before each self-attention layer via ExpandedMemoryCrossAttention.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Backbone: standard ModernBERT (bidirectional)
        self.backbone = ModernBertModel(config)

        # Cross-attention modules (one per layer)
        num_layers = config.num_hidden_layers
        self.cross_attn_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
            for _ in range(num_layers)
        ])
        self.cross_attns = nn.ModuleList([
            ExpandedMemoryCrossAttention(config) for _ in range(num_layers)
        ])

        # Number embedder for decoder input
        self.number_embedder = NumberEmbedder(config)

        # Prediction heads
        self.lm_head = PredictionHead(config)
        self.number_head = NumberHead(config)

    def _get_tok_embeddings(self):
        return self.backbone.embeddings.tok_embeddings

    def forward(self, inputs_embeds, cls_hidden, attention_mask=None):
        """
        Args:
            inputs_embeds: (B, S, D) merged text+number embeddings (pre-norm/drop)
            cls_hidden: (B, D) encoder CLS representation
            attention_mask: (B, S) padding mask (1=valid, 0=pad)
        Returns:
            text_logits: (B, S, vocab_size)
            mag_logits: (B, S, num_magnitude_bins)
        """
        # Apply embedding norm + dropout (same as ModernBertModel)
        hidden_states = self.backbone.embeddings(inputs_embeds=inputs_embeds)

        # Position embeddings (per layer type, as ModernBERT requires)
        seq_len = hidden_states.shape[1]
        device = hidden_states.device
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        position_embeddings = {}
        for layer_type in set(self.config.layer_types):
            position_embeddings[layer_type] = self.backbone.rotary_emb(
                hidden_states, position_ids, layer_type
            )

        # Attention masks (per layer type)
        if not isinstance(attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": hidden_states,
                "attention_mask": attention_mask,
            }
            attention_mask_mapping = {
                "full_attention": create_bidirectional_mask(**mask_kwargs),
                "sliding_attention": create_bidirectional_sliding_window_mask(**mask_kwargs),
            }
        else:
            attention_mask_mapping = attention_mask

        # Run through layers with cross-attention
        for i, layer in enumerate(self.backbone.layers):
            # Cross-attention to CLS (residual)
            hidden_states = hidden_states + self.cross_attns[i](
                self.cross_attn_norms[i](hidden_states), cls_hidden
            )
            # Self-attention + FFN (standard ModernBERT layer)
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask_mapping[layer.attention_type],
                position_embeddings=position_embeddings[layer.attention_type],
            )

        hidden_states = self.backbone.final_norm(hidden_states)

        text_logits = self.lm_head(hidden_states)
        mag_logits = self.number_head(hidden_states)

        return text_logits, mag_logits


class T5StyleModel(nn.Module):
    """Encoder-decoder model for CLS embedding training.

    Encoder: FinancialModernBert (loaded from MLM checkpoint)
    Decoder: CLSBottleneckDecoder (initialized from pretrained ModernBERT)

    Training: encoder sees clean input → CLS token → decoder reconstructs
    masked tokens using CLS + unmasked context.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = FinancialModernBert(config)
        self.decoder = CLSBottleneckDecoder(config)

    def _get_encoder_tok_embeddings(self):
        return self.encoder._get_embedding_layer()

    def _build_embeds(self, input_ids, number_values, is_number_mask, tok_embed_fn, num_embed_fn):
        """Build merged text+number embeddings."""
        text_embeds = tok_embed_fn(input_ids)
        num_embeds = num_embed_fn(number_values)
        mask_expanded = is_number_mask.unsqueeze(-1).expand_as(text_embeds).bool()
        return torch.where(mask_expanded, num_embeds, text_embeds)

    def forward(
        self,
        input_ids,
        number_values,
        is_number_mask,
        attention_mask,
        decoder_input_ids,
        decoder_number_values,
        decoder_is_number_mask,
        labels_text=None,
        labels_magnitude=None,
    ):
        """
        Args:
            input_ids, number_values, is_number_mask, attention_mask:
                Clean encoder inputs.
            decoder_input_ids, decoder_number_values, decoder_is_number_mask:
                Masked decoder inputs (mask_token_id at masked text positions,
                magnitude sentinel at masked number positions).
            labels_text: (B, S) original token IDs at masked text positions, -100 elsewhere.
            labels_magnitude: (B, S) original log-magnitudes at masked number positions, -100 elsewhere.
        """
        # --- Encoder: clean input → CLS ---
        enc_embeds = self._build_embeds(
            input_ids, number_values, is_number_mask,
            self._get_encoder_tok_embeddings(),
            self.encoder.number_embedder,
        )
        encoder_out = self.encoder.modernbert(
            inputs_embeds=enc_embeds, attention_mask=attention_mask
        )
        cls_hidden = encoder_out.last_hidden_state[:, 0, :]  # (B, D)

        # --- Decoder: masked input → predictions ---
        dec_embeds = self._build_embeds(
            decoder_input_ids, decoder_number_values, decoder_is_number_mask,
            self.decoder._get_tok_embeddings(),
            self.decoder.number_embedder,
        )
        text_logits, mag_logits = self.decoder(dec_embeds, cls_hidden, attention_mask)

        # --- Loss ---
        loss = None
        loss_text = None
        loss_mag = None

        if labels_text is not None:
            # Text CE loss (ignores -100 positions automatically)
            loss_text = F.cross_entropy(
                text_logits.view(-1, self.config.vocab_size),
                labels_text.view(-1),
                ignore_index=-100,
            )

            # Magnitude bin loss (same as encoder's NumberHead loss)
            valid_mag_mask = labels_magnitude.view(-1) != -100
            if valid_mag_mask.any():
                target_mags = labels_magnitude.view(-1)[valid_mag_mask]
                pred_mag_logits = mag_logits.view(-1, self.config.num_magnitude_bins)[valid_mag_mask]

                min_v = self.config.magnitude_min
                max_v = self.config.magnitude_max
                n_bins = self.config.num_magnitude_bins

                norm_pos = (target_mags.clamp(min_v, max_v) - min_v) / (max_v - min_v) * (n_bins - 1)
                lower_idx = norm_pos.floor().long()
                upper_idx = norm_pos.ceil().long()
                weight_upper = norm_pos - lower_idx.float()
                weight_lower = 1.0 - weight_upper

                log_probs = F.log_softmax(pred_mag_logits, dim=-1)
                loss_mag = -(
                    weight_lower * log_probs.gather(1, lower_idx.unsqueeze(1)).squeeze(1)
                    + weight_upper * log_probs.gather(1, upper_idx.unsqueeze(1)).squeeze(1)
                )
                loss_mag = loss_mag.mean()
            else:
                loss_mag = torch.tensor(0.0, device=text_logits.device)

            loss = loss_text + loss_mag

        return {
            "loss": loss,
            "loss_text": loss_text,
            "loss_mag": loss_mag,
            "text_logits": text_logits,
            "magnitude_logits": mag_logits,
            "cls_hidden": cls_hidden,
        }


def build_t5_model(
    encoder_checkpoint: str,
    pretrained_model_id: str = "answerdotai/ModernBERT-base",
    num_magnitude_bins: int = 128,
) -> T5StyleModel:
    """Build T5StyleModel with encoder from checkpoint and decoder from pretrained.

    Initialization strategy:
    - Encoder: loaded from MLM training checkpoint (full_model.pt)
    - Decoder backbone: loaded from pretrained ModernBERT
    - Decoder cross-attention: ExpandedMemoryCrossAttention (zero-init on output projection)
    - Decoder number_embedder: copied from encoder
    - Decoder number_head: copied from encoder
    - Decoder lm_head: from pretrained ModernBERT, weight tied to decoder embeddings
    """
    # Config
    config = FinancialModernBertConfig.from_pretrained(pretrained_model_id)
    config.num_magnitude_bins = num_magnitude_bins

    # Build full model
    model = T5StyleModel(config)

    # --- Load encoder from MLM checkpoint ---
    print(f"Loading encoder from {encoder_checkpoint}...")
    ckpt = torch.load(encoder_checkpoint, map_location="cpu", weights_only=False)
    # Handle both raw state_dict and wrapped checkpoint formats
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        encoder_state = ckpt["model_state_dict"]
    else:
        encoder_state = ckpt
    # Strip _orig_mod. prefix from torch.compile'd checkpoints
    encoder_state = {
        k.removeprefix("_orig_mod."): v for k, v in encoder_state.items()
    }
    model.encoder.load_state_dict(encoder_state)

    # --- Load decoder backbone from pretrained ---
    print(f"Loading decoder backbone from {pretrained_model_id}...")
    donor = ModernBertForMaskedLM.from_pretrained(pretrained_model_id)

    model.decoder.backbone.load_state_dict(donor.model.state_dict())

    # Decoder lm_head: dense + norm from donor, decoder weight tied to embeddings
    model.decoder.lm_head.dense.load_state_dict(donor.head.dense.state_dict())
    model.decoder.lm_head.norm.load_state_dict(donor.head.norm.state_dict())
    model.decoder.lm_head.decoder.weight = model.decoder._get_tok_embeddings().weight
    model.decoder.lm_head.decoder.bias.data = donor.decoder.bias.data.clone()

    del donor

    # --- Copy number embedder + head from encoder to decoder ---
    model.decoder.number_embedder.load_state_dict(
        model.encoder.number_embedder.state_dict()
    )
    model.decoder.number_head.load_state_dict(
        model.encoder.number_head.state_dict()
    )

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    print(f"Encoder params: {n_enc/1e6:.1f}M")
    print(f"Decoder params: {n_dec/1e6:.1f}M")
    print(f"Total params:   {(n_enc + n_dec)/1e6:.1f}M")

    return model
