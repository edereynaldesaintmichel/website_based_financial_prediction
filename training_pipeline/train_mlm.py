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
    """Loads pre-padded tensors from a single bucket .pt file.

    Expects the format produced by bucket_by_length.py:
        input_ids:      (N, pad_to)    int32
        is_number_mask: (N, pad_to)    int8
        number_values:  (N, pad_to, 2) float32
        source_files:   [str, ...]
        pad_to:         int
    """

    def __init__(self, pt_path: str, mask_prob: float = 0.15,
                 pad_token_id: int = 0, mask_token_id: int = 50264,
                 magnitude_min: float = -12.0, magnitude_max: float = 12.0):
        data = torch.load(pt_path, map_location="cpu", mmap=True, weights_only=False)
        self.input_ids = data["input_ids"]            # (N, pad_to) int32
        self.is_number_mask = data["is_number_mask"]   # (N, pad_to) int8
        self.number_values = data["number_values"]     # (N, pad_to, 2) float32
        self.source_files = data["source_files"]       # list[str]
        self.pad_to = data["pad_to"]
        self.mask_prob = mask_prob
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.magnitude_min = magnitude_min
        self.magnitude_max = magnitude_max

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx].to(torch.long).clone()
        is_number_mask = self.is_number_mask[idx].float()
        number_values = self.number_values[idx].clone()
        attention_mask = (input_ids != self.pad_token_id).long()
        seq_len = attention_mask.sum().item()

        # Create MLM labels and masked input
        labels_text = torch.full((self.pad_to,), -100, dtype=torch.long)
        labels_sign = torch.full((self.pad_to,), -100, dtype=torch.long)
        labels_magnitude = torch.full((self.pad_to,), -100.0, dtype=torch.float)
        masked_input_ids = input_ids.clone()

        for i in range(seq_len):
            if is_number_mask[i] == 1:
                labels_sign[i] = int(number_values[i, 0].item())
                labels_magnitude[i] = number_values[i, 1].item()
                if random.random() < self.mask_prob:
                    r = random.random()
                    if r < 0.8:
                        # Sentinel: off-distribution value
                        number_values[i, 0] = 0.0
                        number_values[i, 1] = self.magnitude_max
                    elif r < 0.9:
                        # Random number
                        number_values[i, 0] = float(random.randint(0, 1))
                        number_values[i, 1] = random.uniform(self.magnitude_min, self.magnitude_max)
            elif attention_mask[i] == 1:
                if random.random() < self.mask_prob:
                    labels_text[i] = input_ids[i]
                    r = random.random()
                    if r < 0.8:
                        masked_input_ids[i] = self.mask_token_id
                    elif r < 0.9:
                        masked_input_ids[i] = random.randint(0, 50263)

        return {
            "input_ids": masked_input_ids,
            "attention_mask": attention_mask,
            "is_number_mask": is_number_mask,
            "number_values": number_values,
            "labels_text": labels_text,
            "labels_sign": labels_sign,
            "labels_magnitude": labels_magnitude,
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
        bucket_info: {bucket_name: (indices_list, pad_to)}
            indices_list: explicit list of dataset indices for this split
        """
        self.bucket_info = bucket_info
        self.tokens_per_batch = tokens_per_batch
        self.min_batch = min_batch

        # Pre-compute batch sizes and total batch count
        self._total_batches = 0
        self._batch_sizes = {}
        for name, (indices, pad_to) in bucket_info.items():
            bs = compute_batch_size(pad_to, tokens_per_batch, min_batch)
            self._batch_sizes[name] = bs
            self._total_batches += (len(indices) + bs - 1) // bs

    def __iter__(self):
        all_batches = []
        for name, (indices, _pad_to) in self.bucket_info.items():
            bs = self._batch_sizes[name]
            shuffled = list(indices)
            random.shuffle(shuffled)
            for i in range(0, len(shuffled), bs):
                batch = shuffled[i:i + bs]
                all_batches.append(batch)
        random.shuffle(all_batches)
        yield from all_batches

    def __len__(self):
        return self._total_batches

    def summary(self) -> str:
        lines = []
        for name, (indices, pad_to) in self.bucket_info.items():
            bs = self._batch_sizes[name]
            count = len(indices)
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
        modules_to_save=["number_embedder", "number_head"],
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
    # Enable TF32 for any residual FP32 matmuls (Ampere+)
    if device.type == "cuda":
        torch.set_float32_matmul_precision('medium')

    if device.type == "cuda" and args.dtype in ("fp16", "bf16"):
        use_amp = True
        amp_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
        use_scaler = args.dtype == "fp16"  # bf16 has full exponent range, no scaler needed
    else:
        use_amp = False
        amp_dtype = torch.float32
        use_scaler = False
    print(f"Device: {device}, dtype: {args.dtype}" + (" (AMP)" if use_amp else ""))

    # Build model
    print("Building model...")
    model = build_model(args.model_name)

    # Apply LoRA
    print(f"Applying LoRA (rank={args.lora_rank})...")
    model = setup_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
    model = model.to(device)

    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Load tokenizer just for token IDs
    tokenizer = FinancialBertTokenizer(args.model_name)

    # Load all bucket files and create datasets
    bucket_dir = Path(args.data_dir)
    bucket_files = sorted(bucket_dir.glob("bucket_*.pt"))

    if not bucket_files:
        print(f"No bucket .pt files found in {args.data_dir}")
        return

    datasets = []
    bucket_info = {}
    offset = 0

    for bf in bucket_files:
        ds = BucketedMLMDataset(
            str(bf),
            mask_prob=args.mask_prob,
            pad_token_id=tokenizer.pad_token_id or 0,
            mask_token_id=tokenizer.mask_token_id,
        )
        if len(ds) == 0:
            continue
        datasets.append(ds)
        bucket_info[bf.stem] = (offset, len(ds), ds.pad_to)
        offset += len(ds)

    # Concatenate datasets
    combined = torch.utils.data.ConcatDataset(datasets)

    # Three-way split based on source_file extension:
    #   .txt → regularization (Wikipedia etc.)
    #   .md  → financial, further split into train/val by document
    financial_docs = set()
    reg_docs = set()
    for ds in datasets:
        for sf in ds.source_files:
            if sf.endswith(".txt"):
                reg_docs.add(sf)
            else:
                financial_docs.add(sf)

    # Document-level 90/10 split for financial docs only
    val_ratio = 0.1
    financial_docs = sorted(financial_docs)
    random.shuffle(financial_docs)
    val_count = max(1, int(len(financial_docs) * val_ratio)) if financial_docs else 0
    val_docs = set(financial_docs[:val_count])

    print(f"\nDocument split: {len(financial_docs)} financial "
          f"({len(financial_docs) - val_count} train, {val_count} val), "
          f"{len(reg_docs)} regularization")

    # Assign chunks to train/val/reg based on source document
    train_bucket_info = {}
    val_bucket_info = {}
    reg_bucket_info = {}
    for ds_idx, (name, (boffset, count, pad_to)) in enumerate(bucket_info.items()):
        ds = datasets[ds_idx]
        train_indices = []
        val_indices = []
        reg_indices = []
        for local_idx in range(count):
            abs_idx = boffset + local_idx
            sf = ds.source_files[local_idx]
            if sf.endswith(".txt"):
                reg_indices.append(abs_idx)
            elif sf in val_docs:
                val_indices.append(abs_idx)
            else:
                train_indices.append(abs_idx)
        train_bucket_info[name] = (train_indices, pad_to)
        val_bucket_info[name] = (val_indices, pad_to)
        reg_bucket_info[name] = (reg_indices, pad_to)

    batch_sampler = MultiBucketBatchSampler(
        train_bucket_info, args.tokens_per_batch, min_batch=args.min_batch_size,
    )
    val_batch_sampler = MultiBucketBatchSampler(
        val_bucket_info, args.tokens_per_batch, min_batch=args.min_batch_size,
    )

    # Regularization sampler (may be empty if no .txt data present)
    reg_bucket_info_nonempty = {k: v for k, v in reg_bucket_info.items() if len(v[0]) > 0}
    has_reg = len(reg_bucket_info_nonempty) > 0 and args.regularization_ratio > 0
    if has_reg:
        reg_batch_sampler = MultiBucketBatchSampler(
            reg_bucket_info_nonempty, args.tokens_per_batch, min_batch=args.min_batch_size,
        )

    print(f"\nTrain batch plan (tokens_per_batch={args.tokens_per_batch}):")
    print(batch_sampler.summary())
    print(f"Total train batches per epoch: {len(batch_sampler)}")
    print(f"\nVal batch plan:")
    print(val_batch_sampler.summary())
    print(f"Total val batches per epoch: {len(val_batch_sampler)}")
    if has_reg:
        print(f"\nRegularization batch plan (ratio={args.regularization_ratio:.0%}):")
        print(reg_batch_sampler.summary())
        print(f"Total reg batches per epoch: {len(reg_batch_sampler)}")
    print()

    dataloader = DataLoader(
        combined,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=variable_length_collate,
    )
    val_dataloader = DataLoader(
        combined,
        batch_sampler=val_batch_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=variable_length_collate,
    )
    reg_dataloader = None
    if has_reg:
        reg_dataloader = DataLoader(
            combined,
            batch_sampler=reg_batch_sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            collate_fn=variable_length_collate,
        )

    # Optimizer: separate param groups
    lora_params = []
    number_params = []  # number_embedder + number_head (always trainable)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(param)
        elif "number_embedder" in name or "number_head" in name:
            number_params.append(param)

    print(f"LoRA params: {sum(p.numel() for p in lora_params):,}")
    print(f"Number params: {sum(p.numel() for p in number_params):,}")

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr_lora},
        {"params": number_params, "lr": args.lr_heads},
    ], weight_decay=args.weight_decay)

    # AMP scaler — only needed for fp16 (bf16 has full exponent range)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # Warmup: freeze everything except number_embedder + number_head
    if args.warmup_steps > 0:
        print(f"\nWarmup: freezing LoRA for {args.warmup_steps} steps "
              f"(training number embedder/head only)")
        for p in lora_params:
            p.requires_grad = False

    # Training loop
    model.train()
    global_step = 0
    lora_unfrozen = args.warmup_steps == 0

    for epoch in range(args.epochs):
        totals = {"loss": 0.0, "text": 0.0, "sign": 0.0, "mag": 0.0}
        reg_totals = {"loss": 0.0, "count": 0}
        num_batches = 0

        # Fresh regularization iterator each epoch
        reg_iter = iter(reg_dataloader) if reg_dataloader else None

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")
        for batch in pbar:
            # Unfreeze LoRA after warmup
            if not lora_unfrozen and global_step >= args.warmup_steps:
                for p in lora_params:
                    p.requires_grad = True
                lora_unfrozen = True
                print(f"\n  Step {global_step}: unfreezing LoRA")

            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
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

            # Regularization batch (interleaved)
            if reg_iter and random.random() < args.regularization_ratio:
                try:
                    reg_batch = next(reg_iter)
                except StopIteration:
                    reg_iter = iter(reg_dataloader)
                    reg_batch = next(reg_iter)

                reg_batch = {k: v.to(device) for k, v in reg_batch.items()}
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    reg_outputs = model(
                        input_ids=reg_batch["input_ids"],
                        attention_mask=reg_batch["attention_mask"],
                        is_number_mask=reg_batch["is_number_mask"],
                        number_values=reg_batch["number_values"],
                        labels_text=reg_batch["labels_text"],
                        labels_sign=reg_batch["labels_sign"],
                        labels_magnitude=reg_batch["labels_magnitude"],
                    )
                reg_loss = reg_outputs["loss"]
                scaler.scale(reg_loss).backward()
                reg_totals["loss"] += reg_loss.item()
                reg_totals["count"] += 1

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
                reg=f"{reg_totals['loss']/max(1,reg_totals['count']):.4f}" if reg_totals['count'] > 0 else "n/a",
            )

        n = max(num_batches, 1)
        print(f"Epoch {epoch+1}/{args.epochs} — "
              f"train loss: {totals['loss']/n:.4f}  "
              f"text: {totals['text']/n:.4f}  "
              f"sign: {totals['sign']/n:.4f}  "
              f"mag: {totals['mag']/n:.4f}")
        if reg_totals["count"] > 0:
            print(f"  regularization loss: {reg_totals['loss']/reg_totals['count']:.4f} "
                  f"({reg_totals['count']} batches)")

        # Validation
        model.eval()
        val_totals = {"loss": 0.0, "text": 0.0, "sign": 0.0, "mag": 0.0}
        val_batches = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="  Validation", unit="batch"):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        is_number_mask=batch["is_number_mask"],
                        number_values=batch["number_values"],
                        labels_text=batch["labels_text"],
                        labels_sign=batch["labels_sign"],
                        labels_magnitude=batch["labels_magnitude"],
                    )
                val_batches += 1
                val_totals["loss"] += outputs["loss"].item()
                val_totals["text"] += outputs["loss_text"].item()
                val_totals["sign"] += outputs["loss_sign"].item()
                val_totals["mag"] += outputs["loss_mag"].item()

        vn = max(val_batches, 1)
        print(f"Epoch {epoch+1}/{args.epochs} — "
              f"val loss: {val_totals['loss']/vn:.4f}  "
              f"text: {val_totals['text']/vn:.4f}  "
              f"sign: {val_totals['sign']/vn:.4f}  "
              f"mag: {val_totals['mag']/vn:.4f}")
        model.train()

        # Save checkpoint
        if args.save_dir:
            save_path = os.path.join(args.save_dir, f"checkpoint_epoch{epoch+1}")
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)

            # Append train/val losses to the PEFT-generated README.md
            readme_path = os.path.join(save_path, "README.md")
            if os.path.exists(readme_path):
                with open(readme_path, "a") as f:
                    f.write(f"\n## Training Losses (Epoch {epoch+1})\n\n")
                    f.write(f"| Split | Total | Text | Sign | Magnitude |\n")
                    f.write(f"|-------|-------|------|------|-----------|\n")
                    f.write(f"| Train | {totals['loss']/n:.4f} | {totals['text']/n:.4f} "
                            f"| {totals['sign']/n:.4f} | {totals['mag']/n:.4f} |\n")
                    f.write(f"| Val   | {val_totals['loss']/vn:.4f} | {val_totals['text']/vn:.4f} "
                            f"| {val_totals['sign']/vn:.4f} | {val_totals['mag']/vn:.4f} |\n")

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
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--mask_prob", type=float, default=0.15, help="MLM mask probability")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Number of steps to freeze LoRA and train only heads (0 = no warmup)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (e.g. 'cuda', 'cuda:1', 'mps', 'cpu'). "
                             "Auto-detected if not set.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"],
                        help="Training precision: bf16 (recommended for Ampere+/Blackwell), "
                             "fp16 (with loss scaling), or fp32 (no mixed precision)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers for parallel data loading")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for kernel fusion (adds warmup time)")
    parser.add_argument("--regularization_ratio", type=float, default=0,
                        help="Probability of inserting a regularization batch (.txt data) "
                             "after each financial batch (0 = disabled)")
    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()

"""python -m training_pipeline.train_mlm     --data_dir '/content/drive/MyDrive/website predictor/bucketed'     --tokens_per_batch 32768     --dtype bf16     --compile     --num_workers 4     --epochs 3 --lr_heads 3e-4 --lr_lora 3e-4"""


