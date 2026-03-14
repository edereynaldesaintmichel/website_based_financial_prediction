"""
Full-parameter MLM training for FinancialModernBert (no LoRA).

Same data pipeline as train_mlm.py but trains ALL parameters of the
ModernBERT backbone, number embedder, and number head.

To mitigate catastrophic forgetting, regularization data (.txt source files)
is interleaved with financial data (.md source files) at a configurable ratio.
Both go through the same chunk → tokenize → bucket pipeline.

Usage:
    python -m training_pipeline.train_mlm_full \
        --data_dir training_data/bucketed \
        --tokens_per_batch 4096 \
        --epochs 3 \
        --lr 5e-5 \
        --regularization_ratio 0.3
"""
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, Sampler, ConcatDataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from financial_bert import build_model, FinancialBertTokenizer


# ---------------------------------------------------------------------------
# Dataset — bucketed MLM (same as train_mlm.py)
# ---------------------------------------------------------------------------

class BucketedMLMDataset(Dataset):
    """Loads pre-tokenized sequences from a single bucket file."""

    def __init__(self, jsonl_path: str, pad_to: int, mask_prob: float = 0.15,
                 pad_token_id: int = 0, mask_token_id: int = 50264):
        self.items = []
        self.source_files = []
        with open(jsonl_path, "r") as f:
            for line in f:
                item = json.loads(line)
                self.source_files.append(item.get("source_file", ""))
                self.items.append(item)
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
        labels_text = [-100] * self.pad_to
        labels_sign = [-100] * self.pad_to
        labels_magnitude = [-100.0] * self.pad_to
        masked_input_ids = list(input_ids)

        for i in range(seq_len):
            if is_number_mask[i] == 1:
                labels_sign[i] = int(number_values[i][0])
                labels_magnitude[i] = number_values[i][1]
                if random.random() < self.mask_prob:
                    number_values[i] = [0.0, 0.0]
            elif attention_mask[i] == 1:
                if random.random() < self.mask_prob:
                    labels_text[i] = input_ids[i]
                    r = random.random()
                    if r < 0.8:
                        masked_input_ids[i] = self.mask_token_id
                    elif r < 0.9:
                        masked_input_ids[i] = random.randint(0, 50263)

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
# Multi-bucket sampler (same as train_mlm.py)
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


def variable_length_collate(batch):
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "mps" if torch.backends.mps.is_available()
                              else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision('medium')

    if device.type == "cuda" and args.dtype in ("fp16", "bf16"):
        use_amp = True
        amp_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
        use_scaler = args.dtype == "fp16"
    else:
        use_amp = False
        amp_dtype = torch.float32
        use_scaler = False
    print(f"Device: {device}, dtype: {args.dtype}" + (" (AMP)" if use_amp else ""))

    # Build model — full parameter training, no LoRA
    print("Building model (full fine-tuning, no LoRA)...")
    model = build_model(args.model_name)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, trainable: {trainable_params:,}")

    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Load tokenizer
    tokenizer = FinancialBertTokenizer(args.model_name)

    # ------------------------------------------------------------------
    # Load bucketed data
    # ------------------------------------------------------------------
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

    combined = ConcatDataset(datasets)

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

    print(f"\nFinancial train batch plan (tokens_per_batch={args.tokens_per_batch}):")
    print(batch_sampler.summary())
    print(f"Total financial train batches per epoch: {len(batch_sampler)}")
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

    # ------------------------------------------------------------------
    # Optimizer — single param group for full fine-tuning
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Linear warmup + cosine decay schedule
    total_steps = len(batch_sampler) * args.epochs
    warmup_steps = min(args.warmup_steps, total_steps // 5)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    model.train()
    global_step = 0

    for epoch in range(args.epochs):
        totals = {"loss": 0.0, "text": 0.0, "sign": 0.0, "mag": 0.0}
        reg_totals = {"loss": 0.0, "count": 0}
        num_batches = 0

        # Fresh regularization iterator each epoch
        reg_iter = iter(reg_dataloader) if reg_dataloader else None

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            # --- Financial batch ---
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

            # --- Regularization batch (interleaved) ---
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
            scheduler.step()

            num_batches += 1
            global_step += 1
            totals["loss"] += loss.item()
            totals["text"] += outputs["loss_text"].item()
            totals["sign"] += outputs["loss_sign"].item()
            totals["mag"] += outputs["loss_mag"].item()

            pbar.set_postfix(
                loss=f"{totals['loss']/num_batches:.4f}",
                txt=f"{totals['text']/num_batches:.4f}",
                sgn=f"{totals['sign']/num_batches:.4f}",
                mag=f"{totals['mag']/num_batches:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
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

        # Save checkpoint (full model, not PEFT)
        if args.save_dir:
            save_path = os.path.join(args.save_dir, f"checkpoint_epoch{epoch+1}")
            os.makedirs(save_path, exist_ok=True)
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": totals["loss"] / n,
                "val_loss": val_totals["loss"] / vn,
            }, os.path.join(save_path, "full_model.pt"))
            print(f"  Saved to {save_path}")

    print("Training complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Full-parameter MLM training for FinancialModernBert")
    parser.add_argument("--data_dir", required=True, help="Directory with bucketed JSONL files")
    parser.add_argument("--save_dir", default="checkpoints/mlm_full",
                        help="Checkpoint save directory")
    parser.add_argument("--model_name", default="answerdotai/ModernBERT-base", help="Base model")
    parser.add_argument("--tokens_per_batch", type=int, default=4096,
                        help="Target total tokens per batch")
    parser.add_argument("--min_batch_size", type=int, default=1,
                        help="Minimum batch size for any bucket")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate (lower than LoRA — full fine-tuning)")
    parser.add_argument("--mask_prob", type=float, default=0.15)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Linear warmup steps before cosine decay")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for kernel fusion")
    parser.add_argument("--regularization_ratio", type=float, default=0.3,
                        help="Probability of inserting a regularization batch (.txt data) "
                             "after each financial batch (0.3 = ~30%% of steps)")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
