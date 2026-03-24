"""
Training script for T5-style CLS embedding learning.

Encoder sees clean input → CLS. Decoder sees ~50% masked input and
reconstructs masked tokens using CLS + unmasked context.

Supports:
- DDP multi-GPU training
- BF16 mixed precision (Hopper/Blackwell GPUs)
- Gradient accumulation
- Cosine LR schedule with warmup
- Per-epoch chunk-size cycling

Usage (single GPU):
    python -m t5_style_training_pipeline.train \
        --data_dir t5_training_data/bucketed \
        --encoder_checkpoint checkpoints/mlm_full_baseline/checkpoint_epoch3/full_model.pt \
        --output_dir checkpoints/t5_cls

Usage (multi-GPU via torchrun):
    torchrun --nproc_per_node=4 -m t5_style_training_pipeline.train \
        --data_dir t5_training_data/bucketed \
        --encoder_checkpoint checkpoints/mlm_full_baseline/checkpoint_epoch3/full_model.pt \
        --output_dir checkpoints/t5_cls
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from t5_style_training_pipeline.decoder import build_t5_model


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BucketedDataset(Dataset):
    """Loads all bucket_*.pt files and presents them as a flat dataset."""

    def __init__(self, bucket_dir: str):
        self.samples = []  # list of (input_ids, is_number_mask, number_values)
        bucket_files = sorted(Path(bucket_dir).glob("bucket_*.pt"))
        if not bucket_files:
            raise FileNotFoundError(f"No bucket_*.pt files in {bucket_dir}")

        for bf in bucket_files:
            data = torch.load(bf, map_location="cpu", weights_only=False)
            n = data["input_ids"].shape[0]
            for i in range(n):
                self.samples.append({
                    "input_ids": data["input_ids"][i].long(),
                    "is_number_mask": data["is_number_mask"][i].float(),
                    "number_values": data["number_values"][i],
                })
            pad_to = data["pad_to"]
            print(f"  Loaded {bf.name}: {n} seqs, pad_to={pad_to}")

        print(f"  Total: {len(self.samples)} sequences")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Collate with dynamic padding to longest sequence in batch."""
    max_len = max(
        (s["input_ids"] != 0).sum().item() for s in batch
    )
    # Trim to max_len to reduce padding waste
    return {
        "input_ids": torch.stack([s["input_ids"][:max_len] for s in batch]),
        "is_number_mask": torch.stack([s["is_number_mask"][:max_len] for s in batch]),
        "number_values": torch.stack([s["number_values"][:max_len] for s in batch]),
    }


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

    Args:
        input_ids: (B, S) original token IDs
        is_number_mask: (B, S) 1.0 at number positions
        number_values: (B, S) log-magnitudes
        mask_prob: fraction of tokens to mask
        mask_token_id: [MASK] token ID
        pad_token_id: PAD token ID
        magnitude_sentinel: value > magnitude_max to trigger mask bin in NumberEmbedder

    Returns:
        decoder_input_ids, decoder_number_values, decoder_is_number_mask,
        labels_text, labels_magnitude
    """
    B, S = input_ids.shape
    device = input_ids.device

    # Mask candidates: not PAD, not position 0 (CLS)
    is_number = is_number_mask.bool()
    candidates = (input_ids != pad_token_id)
    candidates[:, 0] = False  # never mask CLS

    # Random masking
    rand = torch.rand(B, S, device=device)
    mask_positions = (rand < mask_prob) & candidates

    # Decoder inputs
    decoder_input_ids = input_ids.clone()
    decoder_number_values = number_values.clone()
    decoder_is_number_mask = is_number_mask.clone()

    # Mask text tokens: replace with [MASK]
    text_mask = mask_positions & ~is_number
    decoder_input_ids[text_mask] = mask_token_id

    # Mask number tokens: set magnitude to sentinel (triggers mask bin)
    num_mask = mask_positions & is_number
    decoder_number_values[num_mask] = magnitude_sentinel
    # Also replace input_id so token embedding is [MASK] (gets overridden by
    # number embedding via is_number_mask, but cleaner)
    decoder_input_ids[num_mask] = mask_token_id

    # Labels: only at masked positions, -100 elsewhere
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
# Training
# ---------------------------------------------------------------------------

def train(args):
    # DDP setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = local_rank == 0

    if is_main:
        print(f"World size: {world_size}, Device: {device}")
        print(f"Loading data from {args.data_dir}...")

    # Dataset
    dataset = BucketedDataset(args.data_dir)

    if is_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True
        )
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Model
    if is_main:
        print("\nBuilding model...")
    model = build_t5_model(
        encoder_checkpoint=args.encoder_checkpoint,
        pretrained_model_id=args.pretrained_model_id,
        num_magnitude_bins=args.num_magnitude_bins,
    )
    model = model.to(device)

    # Extract tokenizer constants
    from financial_bert import FinancialBertTokenizer
    tokenizer = FinancialBertTokenizer(args.pretrained_model_id)
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id
    magnitude_sentinel = model.config.magnitude_max + 1.0

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    raw_model = model.module if is_distributed else model

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = len(dataloader) // args.grad_accum_steps
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.amp.GradScaler("cuda", enabled=(args.precision == "fp16"))
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    os.makedirs(args.output_dir, exist_ok=True)

    if is_main:
        print(f"\nTraining: {args.epochs} epochs, {steps_per_epoch} steps/epoch, "
              f"{total_steps} total steps")
        print(f"Batch size: {args.batch_size} x {args.grad_accum_steps} accum "
              f"x {world_size} GPUs = {args.batch_size * args.grad_accum_steps * world_size} effective")
        print(f"LR: {args.lr}, warmup: {warmup_steps} steps, mask prob: {args.mask_prob}")
        print()

    global_step = 0
    for epoch in range(args.epochs):
        if is_distributed:
            sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        epoch_loss_text = 0.0
        epoch_loss_mag = 0.0
        epoch_batches = 0
        t0 = time.time()

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            is_number_mask = batch["is_number_mask"].to(device)
            number_values = batch["number_values"].to(device)
            attention_mask = (input_ids != pad_token_id).long()

            # Create masked decoder inputs
            (
                dec_input_ids,
                dec_number_values,
                dec_is_number_mask,
                labels_text,
                labels_magnitude,
            ) = create_masked_inputs(
                input_ids, is_number_mask, number_values,
                mask_prob=args.mask_prob,
                mask_token_id=mask_token_id,
                pad_token_id=pad_token_id,
                magnitude_sentinel=magnitude_sentinel,
            )

            with torch.amp.autocast("cuda", dtype=amp_dtype):
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

            if args.precision == "fp16":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += outputs["loss"].item()
            epoch_loss_text += outputs["loss_text"].item()
            epoch_loss_mag += outputs["loss_mag"].item()
            epoch_batches += 1

            if (batch_idx + 1) % args.grad_accum_steps == 0:
                if args.precision == "fp16":
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_main and global_step % args.log_every == 0:
                    avg_loss = epoch_loss / epoch_batches
                    avg_text = epoch_loss_text / epoch_batches
                    avg_mag = epoch_loss_mag / epoch_batches
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    throughput = epoch_batches * args.batch_size / elapsed

                    print(
                        f"  step {global_step:>6} | "
                        f"loss {avg_loss:.4f} (text {avg_text:.4f}, mag {avg_mag:.4f}) | "
                        f"lr {lr:.2e} | {throughput:.0f} seq/s"
                    )

        # Epoch summary
        if is_main:
            avg_loss = epoch_loss / max(epoch_batches, 1)
            elapsed = time.time() - t0
            print(f"\nEpoch {epoch + 1}/{args.epochs}: "
                  f"avg loss {avg_loss:.4f}, {elapsed:.0f}s")

            # Save checkpoint
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint_epoch{epoch + 1}")
            os.makedirs(ckpt_dir, exist_ok=True)

            torch.save(raw_model.state_dict(), os.path.join(ckpt_dir, "full_model.pt"))
            torch.save(
                raw_model.encoder.state_dict(),
                os.path.join(ckpt_dir, "encoder.pt"),
            )
            torch.save(
                {"epoch": epoch + 1, "global_step": global_step, "args": vars(args)},
                os.path.join(ckpt_dir, "training_state.pt"),
            )
            print(f"  Saved checkpoint to {ckpt_dir}/\n")

    if is_distributed:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="T5-style CLS embedding training")

    # Data
    parser.add_argument("--data_dir", required=True,
                        help="Directory with bucket_*.pt files")
    parser.add_argument("--encoder_checkpoint", required=True,
                        help="Path to encoder MLM checkpoint (full_model.pt)")

    # Model
    parser.add_argument("--pretrained_model_id", default="answerdotai/ModernBERT-base")
    parser.add_argument("--num_magnitude_bins", type=int, default=128)

    # Training
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--mask_prob", type=float, default=0.5)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")

    # Output
    parser.add_argument("--output_dir", default="checkpoints/t5_cls")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
