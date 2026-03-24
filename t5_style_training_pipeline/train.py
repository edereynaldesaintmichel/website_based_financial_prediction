"""
Training script for T5-style CLS embedding learning.

Encoder sees clean input → CLS. Decoder sees ~50% masked input and
reconstructs masked tokens using CLS + unmasked context.

Supports:
- BF16 mixed precision
- Gradient accumulation
- Cosine LR schedule with warmup
- Dynamic batching via bucketed sampler
- Validation after each epoch
- Resume from checkpoint
- Periodic checkpointing
- torch.compile

Usage:
    python -m t5_style_training_pipeline.train \
        --data_dir t5_training_data \
        --encoder_checkpoint checkpoints/mlm_full_baseline/checkpoint_epoch3/full_model.pt \
        --output_dir checkpoints/t5_cls
"""
import argparse
import math
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler, ConcatDataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from t5_style_training_pipeline.decoder import build_t5_model


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BucketedDataset(Dataset):
    """Indexes directly into pre-padded bucket tensors (no per-sample dicts)."""

    def __init__(self, pt_path: str):
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        self.input_ids = data["input_ids"]            # (N, pad_to) uint16
        self.is_number_mask = data["is_number_mask"]   # (N, pad_to) int8
        self.number_values = data["number_values"]     # (N, pad_to) float32
        self.pad_to = data["pad_to"]

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx].long(),
            "is_number_mask": self.is_number_mask[idx].float(),
            "number_values": self.number_values[idx],
        }


# ---------------------------------------------------------------------------
# Multi-bucket sampler
# ---------------------------------------------------------------------------

def compute_batch_size(bucket_len: int, tokens_per_batch: int, min_batch: int = 1) -> int:
    return max(min_batch, tokens_per_batch // bucket_len)


class MultiBucketBatchSampler(Sampler):
    """Yields batches from the same bucket with dynamic batch sizes."""

    def __init__(self, bucket_info: dict, tokens_per_batch: int, min_batch: int = 1):
        self.bucket_info = bucket_info
        self.tokens_per_batch = tokens_per_batch
        self.min_batch = min_batch
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
                all_batches.append(shuffled[i:i + bs])
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


def collate_fn(batch, pad_token_id: int = 0):
    """Stack samples and trim trailing padding to max actual length."""
    stacked = {key: torch.stack([item[key] for item in batch]) for key in batch[0]}
    # Find rightmost non-pad position across the batch
    non_pad = (stacked["input_ids"] != pad_token_id).any(dim=0)
    if non_pad.any():
        max_len = non_pad.nonzero()[-1].item() + 1
        stacked = {k: v[:, :max_len] for k, v in stacked.items()}
    return stacked


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def create_masked_inputs(
    input_ids: torch.Tensor,
    is_number_mask: torch.Tensor,
    number_values: torch.Tensor,
    mask_prob: float,
    mask_token_id: int,
    pad_token_id: int,
    magnitude_sentinel: float,
):
    """Create masked decoder inputs and labels.

    Returns:
        decoder_input_ids, decoder_number_values, decoder_is_number_mask,
        labels_text, labels_magnitude
    """
    B, S = input_ids.shape
    device = input_ids.device

    is_number = is_number_mask.bool()
    candidates = (input_ids != pad_token_id)
    candidates[:, 0] = False  # never mask CLS

    rand = torch.rand(B, S, device=device)
    mask_positions = (rand < mask_prob) & candidates

    decoder_input_ids = input_ids.clone()
    decoder_number_values = number_values.clone()
    decoder_is_number_mask = is_number_mask.clone()

    text_mask = mask_positions & ~is_number
    decoder_input_ids[text_mask] = mask_token_id

    num_mask = mask_positions & is_number
    decoder_number_values[num_mask] = magnitude_sentinel
    decoder_input_ids[num_mask] = mask_token_id

    labels_text = torch.full_like(input_ids, -100)
    labels_text[text_mask] = input_ids[text_mask]

    labels_magnitude = torch.full(
        (B, S), -100.0, dtype=number_values.dtype, device=device
    )
    labels_magnitude[num_mask] = number_values[num_mask]

    return (
        decoder_input_ids,
        decoder_number_values,
        decoder_is_number_mask,
        labels_text,
        labels_magnitude,
    )


# ---------------------------------------------------------------------------
# LR Schedule
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split(bucket_dir: str):
    """Load all bucket_*.pt files from a directory into a ConcatDataset + bucket_info."""
    bucket_files = sorted(Path(bucket_dir).glob("bucket_*.pt"))
    if not bucket_files:
        return None, {}

    datasets = []
    bucket_info = {}
    offset = 0

    for bf in bucket_files:
        ds = BucketedDataset(str(bf))
        if len(ds) == 0:
            continue
        indices = list(range(offset, offset + len(ds)))
        bucket_info[bf.stem] = (indices, ds.pad_to)
        datasets.append(ds)
        offset += len(ds)
        print(f"  {bf.name}: {len(ds)} seqs, pad_to={ds.pad_to}")

    if not datasets:
        return None, {}

    combined = ConcatDataset(datasets)
    print(f"  Total: {len(combined)} sequences")
    return combined, bucket_info


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_validation(model, val_dataloader, device, pad_token_id, mask_token_id,
                   magnitude_sentinel, mask_prob):
    """Run validation and return average losses."""
    model.eval()
    totals = {"loss": 0.0, "text": 0.0, "mag": 0.0}
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="  Validation", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            is_number_mask = batch["is_number_mask"].to(device)
            number_values = batch["number_values"].to(device)
            attention_mask = (input_ids != pad_token_id).long()

            (dec_input_ids, dec_number_values, dec_is_number_mask,
             labels_text, labels_magnitude) = create_masked_inputs(
                input_ids, is_number_mask, number_values,
                mask_prob=mask_prob,
                mask_token_id=mask_token_id,
                pad_token_id=pad_token_id,
                magnitude_sentinel=magnitude_sentinel,
            )

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    number_values=number_values,
                    is_number_mask=is_number_mask,
                    attention_mask=attention_mask,
                    decoder_input_ids=dec_input_ids,
                    decoder_number_values=dec_number_values,
                    decoder_is_number_mask=dec_is_number_mask,
                    labels_text=labels_text,
                    labels_magnitude=labels_magnitude,
                )

            n_batches += 1
            totals["loss"] += outputs["loss"].item()
            totals["text"] += outputs["loss_text"].item()
            totals["mag"] += outputs["loss_mag"].item()

    model.train()
    n = max(n_batches, 1)
    return {k: v / n for k, v in totals.items()}


def save_checkpoint(path, model, optimizer, scheduler, epoch, global_step,
                    ma=None, val_loss=None, args=None):
    """Save full checkpoint with optimizer/scheduler state for resuming."""
    os.makedirs(path, exist_ok=True)
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    if ma:
        state["train_loss"] = sum(ma["loss"]) / len(ma["loss"]) if ma["loss"] else float("nan")
    if val_loss is not None:
        state["val_loss"] = val_loss
    if args is not None:
        state["args"] = vars(args)
    torch.save(state, os.path.join(path, "full_model.pt"))

    # Also save encoder separately for downstream use
    torch.save(model.encoder.state_dict(), os.path.join(path, "encoder.pt"))
    print(f"  Saved checkpoint to {path}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("medium")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    print(f"\nLoading train data from {train_dir}...")
    train_dataset, train_bucket_info = load_split(train_dir)
    if train_dataset is None:
        print("No training data found!")
        return

    print(f"\nLoading val data from {val_dir}...")
    val_dataset, val_bucket_info = load_split(val_dir)
    has_val = val_dataset is not None and len(val_bucket_info) > 0

    train_sampler = MultiBucketBatchSampler(
        train_bucket_info, args.tokens_per_batch, min_batch=args.min_batch_size,
    )
    print(f"\nTrain batch plan (tokens_per_batch={args.tokens_per_batch}):")
    print(train_sampler.summary())
    print(f"Total train batches per epoch: {len(train_sampler)}")

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    val_loader = None
    if has_val:
        val_sampler = MultiBucketBatchSampler(
            val_bucket_info, args.tokens_per_batch, min_batch=args.min_batch_size,
        )
        print(f"\nVal batch plan:")
        print(val_sampler.summary())
        print(f"Total val batches: {len(val_sampler)}")

        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("\nBuilding model...")
    model = build_t5_model(
        encoder_checkpoint=args.encoder_checkpoint,
        pretrained_model_id=args.pretrained_model_id,
        num_magnitude_bins=args.num_magnitude_bins,
    )
    model = model.to(device)

    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    from financial_bert import FinancialBertTokenizer
    tokenizer = FinancialBertTokenizer(args.pretrained_model_id)
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id
    magnitude_sentinel = model.config.magnitude_max + 1.0

    # ------------------------------------------------------------------
    # Optimizer + schedule
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = len(train_sampler) // args.grad_accum_steps
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    start_epoch = 0
    global_step = 0

    if args.resume_from:
        ckpt_file = os.path.join(args.resume_from, "full_model.pt")
        print(f"\nResuming from {ckpt_file}...")
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
        state_dict = {k.removeprefix("_orig_mod."): v
                      for k, v in ckpt["model_state_dict"].items()}
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        print(f"  Resumed at epoch={start_epoch}, global_step={global_step}")

    print(f"\nTraining: {args.epochs} epochs, {steps_per_epoch} steps/epoch, "
          f"{total_steps} total steps")
    print(f"tokens_per_batch: {args.tokens_per_batch}, grad_accum: {args.grad_accum_steps}")
    print(f"LR: {args.lr}, warmup: {warmup_steps} steps, mask prob: {args.mask_prob}")
    print()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    MA_WINDOW = 100

    for epoch in range(start_epoch, args.epochs):
        model.train()
        ma = {k: deque(maxlen=MA_WINDOW) for k in ("loss", "text", "mag")}
        last_ckpt_time = time.time()

        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            is_number_mask = batch["is_number_mask"].to(device)
            number_values = batch["number_values"].to(device)
            attention_mask = (input_ids != pad_token_id).long()

            (dec_input_ids, dec_number_values, dec_is_number_mask,
             labels_text, labels_magnitude) = create_masked_inputs(
                input_ids, is_number_mask, number_values,
                mask_prob=args.mask_prob,
                mask_token_id=mask_token_id,
                pad_token_id=pad_token_id,
                magnitude_sentinel=magnitude_sentinel,
            )

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    number_values=number_values,
                    is_number_mask=is_number_mask,
                    attention_mask=attention_mask,
                    decoder_input_ids=dec_input_ids,
                    decoder_number_values=dec_number_values,
                    decoder_is_number_mask=dec_is_number_mask,
                    labels_text=labels_text,
                    labels_magnitude=labels_magnitude,
                )

            loss = outputs["loss"] / args.grad_accum_steps
            loss.backward()

            ma["loss"].append(outputs["loss"].item())
            ma["text"].append(outputs["loss_text"].item())
            ma["mag"].append(outputs["loss_mag"].item())

            if (batch_idx + 1) % args.grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            def _ma_avg(d): return sum(d) / len(d) if d else float("nan")
            pbar.set_postfix(
                loss=f"{_ma_avg(ma['loss']):.4f}",
                txt=f"{_ma_avg(ma['text']):.4f}",
                mag=f"{_ma_avg(ma['mag']):.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

            # Periodic checkpoint
            if args.checkpoint_minutes > 0 and time.time() - last_ckpt_time >= args.checkpoint_minutes * 60:
                ckpt_path = os.path.join(args.output_dir, f"checkpoint_step{global_step}")
                save_checkpoint(ckpt_path, model, optimizer, scheduler,
                                epoch, global_step, ma=ma)
                last_ckpt_time = time.time()

        # Epoch summary
        print(f"Epoch {epoch+1}/{args.epochs} (last {MA_WINDOW} batches) — "
              f"loss: {_ma_avg(ma['loss']):.4f}  "
              f"text: {_ma_avg(ma['text']):.4f}  "
              f"mag: {_ma_avg(ma['mag']):.4f}")

        # Validation
        val_loss = None
        if val_loader:
            val_metrics = run_validation(
                model, val_loader, device, pad_token_id, mask_token_id,
                magnitude_sentinel, args.mask_prob,
            )
            val_loss = val_metrics["loss"]
            print(f"  val loss {val_metrics['loss']:.4f} "
                  f"(text {val_metrics['text']:.4f}, mag {val_metrics['mag']:.4f})")

        # End-of-epoch checkpoint
        ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch + 1}")
        save_checkpoint(ckpt_path, model, optimizer, scheduler,
                        epoch + 1, global_step, ma=ma, val_loss=val_loss, args=args)
        print()

    print("Training complete.")


def main():
    parser = argparse.ArgumentParser(description="T5-style CLS embedding training")

    # Data
    parser.add_argument("--data_dir", required=True,
                        help="Directory with train/ and val/ subdirs of bucket_*.pt files")
    parser.add_argument("--encoder_checkpoint", required=True,
                        help="Path to encoder MLM checkpoint (full_model.pt)")

    # Model
    parser.add_argument("--pretrained_model_id", default="answerdotai/ModernBERT-base")
    parser.add_argument("--num_magnitude_bins", type=int, default=128)

    # Training
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--tokens_per_batch", type=int, default=4096,
                        help="Target total tokens per batch (dynamic batch size per bucket)")
    parser.add_argument("--min_batch_size", type=int, default=1,
                        help="Minimum batch size for any bucket")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--mask_prob", type=float, default=0.5)

    # Output / resuming
    parser.add_argument("--output_dir", default="checkpoints/t5_cls")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for kernel fusion")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint directory to resume from")
    parser.add_argument("--checkpoint_minutes", type=int, default=30,
                        help="Save a checkpoint every N minutes (0 to disable)")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
