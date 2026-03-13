"""
FinancialModernBert model architecture.

This module contains the model definitions for FinancialModernBert, a BERT-based model
designed to handle numerical data in financial texts with specialized tokenization
and encoding for numbers (sign and magnitude).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ModernBertModel, ModernBertPreTrainedModel, ModernBertConfig, ModernBertForMaskedLM, AutoTokenizer
from typing import List, Union, Optional, Dict, Any
import re
import math

class FinancialBertTokenizer:
    def __init__(
        self,
        pretrained_model_name_or_path: str = "answerdotai/ModernBERT-base",
        magnitude_min: float = -12.0,
        magnitude_max: float = 12.0,
    ):
        self.base_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.magnitude_min = magnitude_min
        self.magnitude_max = magnitude_max
        self.number_pattern = re.compile(r'<number>(.*?)</number>', re.DOTALL)

        self.pad_token_id = self.base_tokenizer.pad_token_id
        self.mask_token_id = self.base_tokenizer.mask_token_id
        self.unk_token_id = self.base_tokenizer.unk_token_id
        self.number_placeholder_id = self.unk_token_id or 0

    @property
    def vocab_size(self) -> int:
        return len(self.base_tokenizer)

    def parse_number_to_log(self, num_str: str) -> tuple:
        try:
            value = float(num_str.strip())
        except ValueError:
            value = 0.0
        
        sign_idx = 1 if value < 0 else 0
        abs_val = abs(value)
        
        if abs_val < 1e-20:
            log_val = self.magnitude_min
        else:
            log_val = math.log10(abs_val)
            
        log_val = max(self.magnitude_min, min(log_val, self.magnitude_max))
        return (sign_idx, log_val)

    def _tokenize_single(self, text: str, add_special_tokens: bool = True) -> Dict[str, List]:
        numbers_data = []
        for match in self.number_pattern.finditer(text):
            sign_idx, log_val = self.parse_number_to_log(match.group(1))

            numbers_data.append({
                "start": match.start(),
                "end": match.end(),
                "sign_idx": sign_idx,
                "log_val": log_val
            })

        placeholder = "§"
        processed_text = text
        offset = 0
        number_char_positions = []

        for num_info in numbers_data:
            start = num_info["start"] + offset
            end = num_info["end"] + offset
            processed_text = processed_text[:start] + placeholder + processed_text[end:]
            number_char_positions.append(start)
            offset += len(placeholder) - (num_info["end"] - num_info["start"])

        encoded = self.base_tokenizer(
            processed_text,
            add_special_tokens=add_special_tokens,
            return_offsets_mapping=True,
            return_tensors=None
        )

        input_ids = encoded["input_ids"]
        offset_mapping = encoded["offset_mapping"]

        number_token_indices = set()
        for char_pos in number_char_positions:
            for tok_idx, (start, end) in enumerate(offset_mapping):
                if start <= char_pos < end:
                    number_token_indices.add(tok_idx)
                    break

        result_input_ids = []
        is_number_mask = []
        number_values = []

        num_idx = 0
        for tok_idx, token_id in enumerate(input_ids):
            if tok_idx in number_token_indices:
                num_info = numbers_data[num_idx]
                num_idx += 1
                result_input_ids.append(self.number_placeholder_id)
                is_number_mask.append(1)
                number_values.append([
                    float(num_info["sign_idx"]),
                    float(num_info["log_val"])
                ])
            else:
                result_input_ids.append(token_id)
                is_number_mask.append(0)
                number_values.append([0.0, 0.0])

        return {
            "input_ids": result_input_ids,
            "is_number_mask": is_number_mask,
            "number_values": number_values,
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

        batch_results = [self._tokenize_single(t, add_special_tokens) for t in text]

        batch_input_ids = [r["input_ids"] for r in batch_results]
        batch_is_number_mask = [r["is_number_mask"] for r in batch_results]
        batch_number_values = [r["number_values"] for r in batch_results]

        if truncation and max_length:
            for i in range(len(text)):
                if len(batch_input_ids[i]) > max_length:
                    batch_input_ids[i] = batch_input_ids[i][:max_length]
                    batch_is_number_mask[i] = batch_is_number_mask[i][:max_length]
                    batch_number_values[i] = batch_number_values[i][:max_length]

        seq_lengths = [len(ids) for ids in batch_input_ids]

        if padding == "max_length" and max_length:
            pad_to_length = max_length
        elif padding in (True, "longest"):
            pad_to_length = max(seq_lengths)
        else:
            pad_to_length = None

        batch_attention_mask = []
        pad_id = self.pad_token_id or 0

        for i in range(len(text)):
            seq_len = len(batch_input_ids[i])
            attention_mask = [1] * seq_len

            if pad_to_length and seq_len < pad_to_length:
                pad_len = pad_to_length - seq_len
                batch_input_ids[i] += [pad_id] * pad_len
                attention_mask += [0] * pad_len
                batch_is_number_mask[i] += [0] * pad_len
                batch_number_values[i] += [[0.0, 0.0]] * pad_len

            batch_attention_mask.append(attention_mask)

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
                "is_number_mask": torch.tensor(batch_is_number_mask, dtype=torch.float),
                "number_values": torch.tensor(batch_number_values, dtype=torch.float),
            }

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "is_number_mask": batch_is_number_mask,
            "number_values": batch_number_values,
        }

    def decode(self, token_ids, skip_special_tokens: bool = False, **kwargs) -> str:
        """Decode token IDs back to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.base_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)

    def decode_number(
        self,
        sign_logits: torch.Tensor,
        magnitude_logits: torch.Tensor,
        model_config,
        k: int = 5,
    ) -> dict:
        """
        Decode a single number from model outputs using peak_window strategy.
        
        Args:
            sign_logits: Sign logits from model (shape: [2])
            magnitude_logits: Magnitude logits from model (shape: [num_bins])
            model_config: Model config with magnitude_min, magnitude_max, num_magnitude_bins
            k: Window size for peak_window strategy
        
        Returns:
            Dict with 'value', 'sign', 'log_magnitude'
        """
        device = magnitude_logits.device
        
        # Decode sign
        pred_sign_class = torch.argmax(sign_logits).item()
        sign_val = -1.0 if pred_sign_class == 1 else 1.0
        
        # Decode magnitude using peak_window strategy
        mag_probs = torch.softmax(magnitude_logits, dim=-1)
        num_bins = model_config.num_magnitude_bins
        bin_indices = torch.arange(num_bins, device=device).float()
        
        peak_idx = torch.argmax(mag_probs).item()
        radius = (k - 1) // 2
        start_idx = max(0, peak_idx - radius)
        end_idx = min(num_bins, peak_idx + radius + 1)
        
        window_probs = mag_probs[start_idx:end_idx]
        window_indices = bin_indices[start_idx:end_idx]
        
        # Re-normalize within the window
        window_probs = window_probs / window_probs.sum()
        expected_bin_idx = (window_probs * window_indices).sum().item()
        
        # Convert bin index to log value
        min_v = model_config.magnitude_min
        max_v = model_config.magnitude_max
        pred_log_val = (expected_bin_idx / (num_bins - 1)) * (max_v - min_v) + min_v
        final_value = sign_val * (10.0 ** pred_log_val)
        
        return {
            'value': final_value,
            'sign': pred_sign_class,
            'log_magnitude': pred_log_val,
        }

    def decode_numbers(
        self,
        sign_logits: torch.Tensor,
        magnitude_logits: torch.Tensor,
        is_number_mask: torch.Tensor,
        model_config,
        k: int = 5,
    ) -> list:
        """
        Decode all numbers in a sequence from model outputs.
        
        Args:
            sign_logits: Sign logits from model (shape: [seq_len, 2])
            magnitude_logits: Magnitude logits from model (shape: [seq_len, num_bins])
            is_number_mask: Mask indicating number positions (shape: [seq_len])
            model_config: Model config with magnitude_min, magnitude_max, num_magnitude_bins
            k: Window size for peak_window strategy
        
        Returns:
            List of dicts, one per number in sequence order, each with:
                - 'token_position': position in the tokenized sequence
                - 'value': decoded numeric value
                - 'sign': predicted sign class (0=positive, 1=negative)
                - 'log_magnitude': predicted log10 magnitude
        """
        results = []
        
        # Find all number positions
        number_positions = (is_number_mask == 1).nonzero(as_tuple=True)[0].tolist()
        
        for pos in number_positions:
            decoded = self.decode_number(
                sign_logits[pos],
                magnitude_logits[pos],
                model_config,
                k=k,
            )
            decoded['token_position'] = pos
            results.append(decoded)
        
        return results

    def log_to_value(self, sign_idx: int, log_val: float) -> float:
        """Convert sign index and log value back to original number."""
        sign = -1.0 if sign_idx == 1 else 1.0
        if log_val < -11:  # Effectively zero
            return 0.0
        return sign * (10.0 ** log_val)

class FinancialModernBertConfig(ModernBertConfig):
    def __init__(
        self,
        num_magnitude_bins=128,
        magnitude_min=-12.0,
        magnitude_max=12.0,
        sign_embed_dim=8,
        magnitude_embed_dim=64,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_magnitude_bins = num_magnitude_bins
        self.magnitude_min = magnitude_min
        self.magnitude_max = magnitude_max
        self.sign_embed_dim = sign_embed_dim
        self.magnitude_embed_dim = magnitude_embed_dim

class PredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(config.hidden_size, bias=False, elementwise_affine=True, eps=1e-5)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class GatedNumberEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sign_emb = nn.Embedding(2, config.sign_embed_dim)
        self.magnitude_emb = nn.Embedding(config.num_magnitude_bins, config.magnitude_embed_dim)
        
        input_dim = config.sign_embed_dim + config.magnitude_embed_dim
        self.gate_proj = nn.Linear(input_dim, config.hidden_size)
        self.val_proj = nn.Linear(input_dim, config.hidden_size)
        self.out_norm = nn.LayerNorm(config.hidden_size)

    def _get_magnitude_embeddings(self, log_vals):
        min_v = self.config.magnitude_min
        max_v = self.config.magnitude_max
        n_bins = self.config.num_magnitude_bins
        
        clamped = log_vals.clamp(min_v, max_v)
        norm_pos = (clamped - min_v) / (max_v - min_v) * (n_bins - 1)
        
        lower_idx = norm_pos.floor().long()
        upper_idx = norm_pos.ceil().long()
        
        weight_upper = norm_pos - lower_idx.float()
        weight_lower = 1.0 - weight_upper
        
        emb_lower = self.magnitude_emb(lower_idx)
        emb_upper = self.magnitude_emb(upper_idx)
        
        return emb_lower * weight_lower.unsqueeze(-1) + emb_upper * weight_upper.unsqueeze(-1)

    def forward(self, number_values):
        sign_ids = number_values[..., 0].long()
        log_vals = number_values[..., 1]

        s_emb = self.sign_emb(sign_ids)
        m_emb = self._get_magnitude_embeddings(log_vals)

        concat_feats = torch.cat([s_emb, m_emb], dim=-1)

        gate = torch.sigmoid(self.gate_proj(concat_feats))
        val = self.val_proj(concat_feats)

        out = gate * val
        return self.out_norm(out)

class GatedNumberHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.val_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.gate_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.decoder_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.num_magnitude_bins + 2),
        )

    def forward(self, sequence_output):
        # gate = torch.sigmoid(self.gate_proj(sequence_output))
        # val = self.val_proj(sequence_output)
        # gated_out = gate * val
        
        logits = self.decoder_mlp(sequence_output)
        
        return logits[...,:2], logits[...,2:]

class FinancialModernBert(ModernBertPreTrainedModel):
    config_class = FinancialModernBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.modernbert = ModernBertModel(config)
        self.number_embedder = GatedNumberEmbedder(config)
        self.lm_head = PredictionHead(config)
        self.number_head = GatedNumberHead(config)
        self.post_init()

    def _get_embedding_layer(self):
        embeddings = self.modernbert.embeddings
        if hasattr(embeddings, 'tok_embeddings'):
            return embeddings.tok_embeddings
        elif hasattr(embeddings, 'word_embeddings'):
            return embeddings.word_embeddings
        return embeddings.tok_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        number_values: torch.Tensor,
        is_number_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels_text: Optional[torch.Tensor] = None,
        labels_magnitude: Optional[torch.Tensor] = None,
        labels_sign: Optional[torch.Tensor] = None
    ):
        text_embeds = self._get_embedding_layer()(input_ids)
        num_embeds = self.number_embedder(number_values)

        hidden_dim = self.config.hidden_size
        mask_expanded = is_number_mask.unsqueeze(-1).expand(-1, -1, hidden_dim).bool()

        final_inputs_embeds = torch.where(mask_expanded, num_embeds, text_embeds)

        outputs = self.modernbert(
            inputs_embeds=final_inputs_embeds,
            attention_mask=attention_mask
        )
        sequence_output = outputs.last_hidden_state
        
        text_logits = self.lm_head(sequence_output)
        sign_logits, magnitude_logits = self.number_head(sequence_output)

        loss = None
        if labels_text is not None:
            text_mask = (is_number_mask == 0)
            num_mask = (is_number_mask == 1)

            loss_fct_text = nn.CrossEntropyLoss()
            active_text_logits = text_logits.view(-1, self.config.vocab_size)
            active_text_labels = labels_text.view(-1)
            active_text_labels = torch.where(text_mask.view(-1), active_text_labels, torch.tensor(-100).to(labels_text.device))
            loss_text = loss_fct_text(active_text_logits, active_text_labels) if (active_text_labels != -100).any() else torch.tensor(0.0, device=labels_text.device)

            active_sign_logits = sign_logits.view(-1, 2)
            active_sign_labels = labels_sign.view(-1)
            active_sign_labels = torch.where(num_mask.view(-1), active_sign_labels, torch.tensor(-100).to(labels_sign.device))
            loss_sign = loss_fct_text(active_sign_logits, active_sign_labels) if (active_sign_labels != -100).any() else torch.tensor(0.0, device=labels_sign.device)

            valid_mag_mask = (num_mask.view(-1)) & (labels_magnitude.view(-1) != -100)
            if valid_mag_mask.any():
                target_mags = labels_magnitude.view(-1)[valid_mag_mask]
                pred_mag_logits = magnitude_logits.view(-1, self.config.num_magnitude_bins)[valid_mag_mask]
                
                min_v = self.config.magnitude_min
                max_v = self.config.magnitude_max
                n_bins = self.config.num_magnitude_bins
                
                norm_pos = (target_mags.clamp(min_v, max_v) - min_v) / (max_v - min_v) * (n_bins - 1)
                lower_idx = norm_pos.floor().long()
                upper_idx = norm_pos.ceil().long()
                
                weight_upper = norm_pos - lower_idx.float()
                weight_lower = 1.0 - weight_upper
                
                log_probs = F.log_softmax(pred_mag_logits, dim=-1)
                
                loss_mag = -(weight_lower * log_probs.gather(1, lower_idx.unsqueeze(1)).squeeze(1) + 
                             weight_upper * log_probs.gather(1, upper_idx.unsqueeze(1)).squeeze(1))
                loss_mag = loss_mag.mean()
            else:
                loss_mag = torch.tensor(0.0, device=labels_magnitude.device)

            loss = loss_text + loss_sign + loss_mag

        return {
            "loss": loss,
            "loss_text": loss_text if loss is not None else None,
            "loss_sign": loss_sign if loss is not None else None,
            "loss_mag": loss_mag if loss is not None else None,
            "text_logits": text_logits,
            "sign_logits": sign_logits,
            "magnitude_logits": magnitude_logits
        }

def build_model(model_id="answerdotai/ModernBERT-base", num_magnitude_bins=128):
    donor_model = ModernBertForMaskedLM.from_pretrained(model_id)
    config = FinancialModernBertConfig.from_pretrained(model_id)
    config.num_magnitude_bins = num_magnitude_bins
    
    financial_model = FinancialModernBert(config)
    financial_model.modernbert.load_state_dict(donor_model.model.state_dict())
    financial_model.lm_head.dense.load_state_dict(donor_model.head.dense.state_dict())
    financial_model.lm_head.norm.load_state_dict(donor_model.head.norm.state_dict())

    embedding_layer = financial_model._get_embedding_layer()
    financial_model.lm_head.decoder.weight = embedding_layer.weight
    financial_model.lm_head.decoder.bias.data = donor_model.decoder.bias.data.clone()

    return financial_model
