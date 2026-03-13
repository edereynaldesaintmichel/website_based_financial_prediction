"""
MLM training script for FinancialModernBert with LoRA.

Loads pre-tokenized, bucketed data and trains with:
- LoRA adapters on the ModernBERT backbone (rank 8)
- Full-parameter training on GatedNumberEmbedder + GatedNumberHead
- Dual loss: text MLM + number prediction (sign CE + soft-label magnitude CE)
- Dynamic batch size: inversely proportional to sequence length, keeping
  total tokens per batch roughly constant

Usage:
    python -m training_pipeline.train_mlm \
        --data_dir training_data/bucketed_demo \
        --tokens_per_batch 4096 \
        --epochs 3 \
        --lr_lora 2e-4 \
        --lr_heads 1e-4
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from financial_bert import build_model, FinancialBertTokenizer


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BucketedMLMDataset(Dataset):
    """Loads pre-tokenized sequences from a single bucket file."""

    def __init__(self, jsonl_path: str, pad_to: int, mask_prob: float = 0.15,
                 pad_token_id: int = 0, mask_token_id: int = 50264):
        self.items = []
        with open(jsonl_path, "r") as f:
            for line in f:
                self.items.append(json.loads(line))
        self.pad_to = pad_to
        self.mask_prob = mask_prob
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        input_ids = list(item["input_ids"])
        is_number_mask = list(item["is_number_mask"])
        number_values = [list(v) for v in item["number_values"]]
        seq_len = len(input_ids)

        # Pad to bucket length
        pad_len = self.pad_to - seq_len
        attention_mask = [1] * seq_len + [0] * pad_len
        input_ids = input_ids + [self.pad_token_id] * pad_len
        is_number_mask = is_number_mask + [0] * pad_len
        number_values = number_values + [[0.0, 0.0]] * pad_len

        # Create MLM labels and masked input
        labels_text = list(input_ids)  # copy
        labels_sign = [-100] * self.pad_to
        labels_magnitude = [-100.0] * self.pad_to
        masked_input_ids = list(input_ids)

        for i in range(seq_len):
            if is_number_mask[i] == 1:
                # Number positions: always predict (like masking), but we
                # mask the number embedding by zeroing number_values with prob mask_prob
                labels_sign[i] = int(number_values[i][0])
                labels_magnitude[i] = number_values[i][1]
                labels_text[i] = -100  # don't compute text loss on number positions
                if random.random() < self.mask_prob:
                    # Zero out the number values so the model must predict
                    number_values[i] = [0.0, 0.0]
            elif attention_mask[i] == 1:
                # Text positions: standard MLM masking
                if random.random() < self.mask_prob:
                    # 80% mask, 10% random, 10% keep
                    r = random.random()
                    if r < 0.8:
                        masked_input_ids[i] = self.mask_token_id
                    elif r < 0.9:
                        masked_input_ids[i] = random.randint(0, 50263)  # random token
                    # else: keep original
                    # labels_text[i] already has the original token id
                else:
                    labels_text[i] = -100  # don't compute loss on unmasked
            else:
                labels_text[i] = -100  # padding

        return {
            "input_ids": torch.tensor(masked_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "is_number_mask": torch.tensor(is_number_mask, dtype=torch.float),
            "number_values": torch.tensor(number_values, dtype=torch.float),
            "labels_text": torch.tensor(labels_text, dtype=torch.long),
            "labels_sign": torch.tensor(labels_sign, dtype=torch.long),
            "labels_magnitude": torch.tensor(labels_magnitude, dtype=torch.float),
        }


# ---------------------------------------------------------------------------
# Multi-bucket sampler with dynamic batch size
# ---------------------------------------------------------------------------

def compute_batch_size(bucket_len: int, tokens_per_batch: int, min_batch: int = 1) -> int:
    """
    Compute batch size for a bucket so that total tokens ≈ tokens_per_batch.

    Memory for a transformer forward pass scales as:
        activations ∝ batch_size × seq_len × hidden_size
        attention    ∝ batch_size × num_heads × seq_len² (without flash attn)

    ModernBERT uses flash attention, so the dominant term is the linear one.
    We keep batch_size × seq_len ≈ constant (= tokens_per_batch).
    """
    return max(min_batch, tokens_per_batch // bucket_len)


class MultiBucketBatchSampler(Sampler):
    """
    Yields batches where all items come from the same bucket.
    Batch size varies per bucket to keep total tokens roughly constant.
    Shuffles within and across buckets each epoch.
    """

    def __init__(self, bucket_info: dict, tokens_per_batch: int, min_batch: int = 1):
        """
        bucket_info: {bucket_name: (offset, count, pad_to)}
        """
        self.bucket_info = bucket_info
        self.tokens_per_batch = tokens_per_batch
        self.min_batch = min_batch

        # Pre-compute batch sizes and total batch count
        self._total_batches = 0
        self._batch_sizes = {}
        for name, (_offset, count, pad_to) in bucket_info.items():
            bs = compute_batch_size(pad_to, tokens_per_batch, min_batch)
            self._batch_sizes[name] = bs
            self._total_batches += (count + bs - 1) // bs

    def __iter__(self):
        all_batches = []
        for name, (offset, count, _pad_to) in self.bucket_info.items():
            bs = self._batch_sizes[name]
            indices = list(range(offset, offset + count))
            random.shuffle(indices)
            for i in range(0, len(indices), bs):
                batch = indices[i:i + bs]
                all_batches.append(batch)
        random.shuffle(all_batches)
        yield from all_batches

    def __len__(self):
        return self._total_batches

    def summary(self) -> str:
        lines = []
        for name, (offset, count, pad_to) in self.bucket_info.items():
            bs = self._batch_sizes[name]
            n_batches = (count + bs - 1) // bs
            lines.append(f"  {name}: pad_to={pad_to}, batch_size={bs}, "
                         f"~{bs * pad_to} tokens/batch, {n_batches} batches")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def setup_lora(model, rank=8, alpha=16):
    """Apply LoRA adapters to the ModernBERT backbone."""
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["Wqkv", "Wo", "Wi"],  # ModernBERT attention + MLP projections
        lora_dropout=0.05,
        bias="none",
        modules_to_save=["number_embedder", "number_head", "lm_head"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train(args):
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "mps" if torch.backends.mps.is_available()
                              else "cpu")
    use_amp = device.type == "cuda" and args.amp
    print(f"Device: {device}" + (" (AMP enabled)" if use_amp else ""))

    # Build model
    print("Building model...")
    model = build_model(args.model_name)

    # Apply LoRA
    print(f"Applying LoRA (rank={args.lora_rank})...")
    model = setup_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
    model = model.to(device)

    # Load tokenizer just for token IDs
    tokenizer = FinancialBertTokenizer(args.model_name)

    # Load all bucket files and create datasets
    bucket_dir = Path(args.data_dir)
    bucket_files = sorted(bucket_dir.glob("bucket_*.jsonl"))

    if not bucket_files:
        print(f"No bucket files found in {args.data_dir}")
        return

    datasets = []
    bucket_info = {}
    offset = 0

    for bf in bucket_files:
        bucket_bound = int(bf.stem.split("_")[1])
        ds = BucketedMLMDataset(
            str(bf),
            pad_to=bucket_bound,
            mask_prob=args.mask_prob,
            pad_token_id=tokenizer.pad_token_id or 0,
            mask_token_id=tokenizer.mask_token_id,
        )
        if len(ds) == 0:
            continue
        datasets.append(ds)
        bucket_info[bf.stem] = (offset, len(ds), bucket_bound)
        offset += len(ds)

    # Concatenate datasets
    combined = torch.utils.data.ConcatDataset(datasets)
    batch_sampler = MultiBucketBatchSampler(
        bucket_info, args.tokens_per_batch, min_batch=args.min_batch_size,
    )

    print(f"\nBatch plan (tokens_per_batch={args.tokens_per_batch}):")
    print(batch_sampler.summary())
    print(f"Total batches per epoch: {len(batch_sampler)}\n")

    dataloader = DataLoader(
        combined,
        batch_sampler=batch_sampler,
        num_workers=0,
        collate_fn=variable_length_collate,
    )

    # Optimizer: separate LR for LoRA params vs head params
    lora_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(param)
        else:
            head_params.append(param)

    print(f"LoRA params: {sum(p.numel() for p in lora_params):,}")
    print(f"Head params: {sum(p.numel() for p in head_params):,}")

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr_lora},
        {"params": head_params, "lr": args.lr_heads},
    ], weight_decay=args.weight_decay)

    # AMP scaler for CUDA mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Warmup: freeze LoRA params for the first N steps
    if args.warmup_steps > 0:
        print(f"\nWarmup: freezing LoRA params for {args.warmup_steps} steps "
              f"(training heads only)")
        for p in lora_params:
            p.requires_grad = False

    # Training loop
    model.train()
    global_step = 0
    lora_unfrozen = args.warmup_steps == 0

    for epoch in range(args.epochs):
        totals = {"loss": 0.0, "text": 0.0, "sign": 0.0, "mag": 0.0}
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")
        for batch in pbar:
            # Unfreeze LoRA after warmup
            if not lora_unfrozen and global_step >= args.warmup_steps:
                for p in lora_params:
                    p.requires_grad = True
                lora_unfrozen = True
                print(f"\n  Step {global_step}: unfreezing LoRA params")

            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    is_number_mask=batch["is_number_mask"],
                    number_values=batch["number_values"],
                    labels_text=batch["labels_text"],
                    labels_sign=batch["labels_sign"],
                    labels_magnitude=batch["labels_magnitude"],
                )

            loss = outputs["loss"]
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Track individual losses
            num_batches += 1
            global_step += 1
            totals["loss"] += loss.item()
            totals["text"] += outputs["loss_text"].item()
            totals["sign"] += outputs["loss_sign"].item()
            totals["mag"] += outputs["loss_mag"].item()

            phase = "warmup" if not lora_unfrozen else "full"
            pbar.set_postfix(
                loss=f"{totals['loss']/num_batches:.4f}",
                txt=f"{totals['text']/num_batches:.4f}",
                sgn=f"{totals['sign']/num_batches:.4f}",
                mag=f"{totals['mag']/num_batches:.4f}",
                phase=phase,
            )

        n = max(num_batches, 1)
        print(f"Epoch {epoch+1}/{args.epochs} — "
              f"loss: {totals['loss']/n:.4f}  "
              f"text: {totals['text']/n:.4f}  "
              f"sign: {totals['sign']/n:.4f}  "
              f"mag: {totals['mag']/n:.4f}")

        # Save checkpoint
        if args.save_dir:
            save_path = os.path.join(args.save_dir, f"checkpoint_epoch{epoch+1}")
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            print(f"  Saved to {save_path}")

    print("Training complete.")


def variable_length_collate(batch):
    """
    Collate function that handles variable-length sequences across buckets.
    Within a batch all sequences have the same pad_to length (same bucket),
    so we can just stack normally.
    """
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}


def main():
    parser = argparse.ArgumentParser(description="MLM training with LoRA for FinancialModernBert")
    parser.add_argument("--data_dir", required=True, help="Directory with bucketed JSONL files")
    parser.add_argument("--save_dir", default="checkpoints/mlm_lora", help="Checkpoint save directory")
    parser.add_argument("--model_name", default="answerdotai/ModernBERT-base", help="Base model")
    parser.add_argument("--tokens_per_batch", type=int, default=4096,
                        help="Target total tokens per batch (batch_size = tokens_per_batch // seq_len)")
    parser.add_argument("--min_batch_size", type=int, default=1, help="Minimum batch size for any bucket")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr_lora", type=float, default=2e-4, help="Learning rate for LoRA params")
    parser.add_argument("--lr_heads", type=float, default=1e-4, help="Learning rate for head params")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--mask_prob", type=float, default=0.15, help="MLM mask probability")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Number of steps to freeze LoRA and train only heads (0 = no warmup)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (e.g. 'cuda', 'cuda:1', 'mps', 'cpu'). "
                             "Auto-detected if not set.")
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed-precision training (CUDA only)")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
