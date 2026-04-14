"""
Training script for growth rate prediction from document embeddings.

The CLS embeddings are pre-computed and cached in the HuggingFace dataset
`edereynal/financial_prediction` under `aggregator/cls_cache/` (produced by
cls_aggregator_training_pipeline/train.py). Cache files are:

    cls_cache_{train,val}_epoch{0..N-1}.pt

Each file is a dict: {doc_idx -> Tensor(n_chunks, D)} where `doc_idx` is the
position of the document in the corresponding split of `documents.pt`, with
the same `split_utils.split_documents` used by both pipelines.

Pipeline:
1. Load documents.pt + growth_rates.json, apply the same train/val split as
   the aggregator pipeline, and build a {doc_idx -> growth_rate} table per
   split (by extracting the company_id from each document's source_file).
2. Each epoch: load (or download from HF) the epoch's CLS cache, align with
   growth rates, form batches, forward through GrowthPredictor.
3. Train with MSE + optional aggregator-output regularization.

Usage:
    python -m final_training_pipeline.train \\
        --data mlm_data/documents.pt \\
        --growth-rates growth_rates.json \\
        --aggregator-checkpoint checkpoints/cls_aggregator/checkpoint_epoch10/aggregator.pt
"""
import argparse
import json
import math
import os
import random
import sys
import time
from collections import deque

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from split_utils import split_documents
from final_training_pipeline.model import GrowthPredictor
from cls_aggregator_training_pipeline.aggregator import CLSAggregator


HF_REPO_ID = "edereynal/financial_prediction"
HF_REPO_TYPE = "dataset"
HF_CACHE_PREFIX = "aggregator/cls_cache"
NUM_CACHED_EPOCHS = 5  # matches cls_aggregator_training_pipeline default


# ---------------------------------------------------------------------------
# Company ID extraction (must match the cls_aggregator pipeline filtering)
# ---------------------------------------------------------------------------

def extract_company_id(source_file):
    """SEC: '1000045_2018-06-27.md' -> '1000045'
    UK:  'Prod224_0052_00781277_20171231.md' -> '00781277'
    Wiki: '000000_alabama.txt' -> None"""
    if source_file.endswith('.txt'):
        return None
    if source_file.startswith('Prod'):
        parts = source_file.split('_')
        if len(parts) >= 4:
            return parts[2]
        return None
    return source_file.split('_')[0]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_predictor(aggregator_checkpoint, hidden_size, train_aggregator,
                    num_heads=16, num_layers=6, ffn_mult=4, dropout=0.0):
    """Build GrowthPredictor and load aggregator weights."""
    aggregator = CLSAggregator(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_mult=ffn_mult,
        dropout=dropout,
    )

    if aggregator_checkpoint:
        ckpt = torch.load(aggregator_checkpoint, map_location="cpu",
                          weights_only=False)
        if isinstance(ckpt, dict) and "aggregator" in ckpt:
            state_dict = ckpt["aggregator"]
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
        cleaned = {}
        for k, v in state_dict.items():
            k = k.removeprefix("_orig_mod.")
            k = k.removeprefix("aggregator.")
            cleaned[k] = v
        aggregator.load_state_dict(cleaned)
        print(f"  Loaded aggregator from {aggregator_checkpoint}")

    if not train_aggregator:
        aggregator.eval()
        for p in aggregator.parameters():
            p.requires_grad_(False)
        print("  Aggregator: frozen")
    else:
        print("  Aggregator: trainable")

    predictor = GrowthPredictor(aggregator=aggregator, hidden_size=hidden_size)
    n_head = sum(p.numel() for p in predictor.head.parameters()) / 1e6
    n_agg = sum(p.numel() for p in predictor.aggregator.parameters()) / 1e6
    print(f"  GrowthPredictor: aggregator {n_agg:.1f}M, head {n_head:.2f}M")
    return predictor


# ---------------------------------------------------------------------------
# CLS cache: local + HF fallback
# ---------------------------------------------------------------------------

def cache_filename(split, epoch_idx):
    return f"cls_cache_{split}_epoch{epoch_idx}.pt"


def ensure_cache(output_dir, split, epoch_idx):
    """Return local path to cache file, downloading from HF if missing."""
    fname = cache_filename(split, epoch_idx)
    local = os.path.join(output_dir, fname)
    if os.path.exists(local):
        return local

    from huggingface_hub import hf_hub_download
    repo_path = f"{HF_CACHE_PREFIX}/{fname}"
    print(f"  Downloading {repo_path} from HF ({HF_REPO_ID})...")
    downloaded = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        filename=repo_path,
    )
    # Symlink into output_dir for quick subsequent lookups.
    os.makedirs(output_dir, exist_ok=True)
    try:
        if os.path.islink(local) or os.path.exists(local):
            os.remove(local)
        os.symlink(downloaded, local)
    except OSError:
        pass
    return downloaded if not os.path.exists(local) else local


def load_cls_cache(output_dir, split, epoch_idx):
    """Load a pre-computed CLS cache file. Returns dict[doc_idx -> Tensor(n, D)]."""
    path = ensure_cache(output_dir, split, epoch_idx)
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Align cache with growth rates
# ---------------------------------------------------------------------------

def build_idx_to_rate(split_docs, growth_rates):
    """Map {doc_idx (pos in split) -> growth_rate} for matched, non-wiki docs."""
    idx_to_rate = {}
    for i, doc in enumerate(split_docs):
        cid = extract_company_id(doc["source_file"])
        if cid is None:
            continue
        rate = growth_rates.get(cid)
        if rate is None:
            continue
        idx_to_rate[i] = rate
    return idx_to_rate


def align_cache_with_rates(cls_cache, idx_to_rate):
    """Intersect cache keys with rated docs; return ordered lists."""
    keys = sorted(k for k in cls_cache if k in idx_to_rate)
    doc_cls = [cls_cache[k] for k in keys]
    doc_rates = [idx_to_rate[k] for k in keys]
    return doc_cls, doc_rates, keys


# ---------------------------------------------------------------------------
# Document batching
# ---------------------------------------------------------------------------

def form_doc_batches(doc_cls, doc_rates, cls_budget, shuffle=True):
    """Greedily pack up to cls_budget total CLS tokens per batch."""
    indices = sorted(range(len(doc_cls)), key=lambda i: doc_cls[i].shape[0])

    batches_indices = []
    current_batch = []
    current_total = 0

    for idx in indices:
        n = doc_cls[idx].shape[0]
        if current_batch and current_total + n > cls_budget:
            batches_indices.append(current_batch)
            current_batch = [idx]
            current_total = n
        else:
            current_batch.append(idx)
            current_total += n
    if current_batch:
        batches_indices.append(current_batch)

    if shuffle:
        random.shuffle(batches_indices)

    batches = []
    for batch_idx in batches_indices:
        batch_cls = [doc_cls[i] for i in batch_idx]
        batch_rates = [doc_rates[i] for i in batch_idx]

        max_n = max(t.shape[0] for t in batch_cls)
        D = batch_cls[0].shape[1]
        B = len(batch_cls)

        cls_tokens = torch.zeros(B, max_n, D)
        attention_mask = torch.zeros(B, max_n, dtype=torch.long)

        for i, t in enumerate(batch_cls):
            n = t.shape[0]
            cls_tokens[i, :n] = t
            attention_mask[i, :n] = 1

        targets = torch.tensor(batch_rates, dtype=torch.float32)
        batches.append((cls_tokens, attention_mask, targets))

    return batches


# ---------------------------------------------------------------------------
# Regularization reference outputs
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_reference_outputs(predictor, doc_cls, doc_rates, cls_budget, device):
    """Pre-compute aggregator (mean-pooled) outputs as regularization targets.

    Returns dict: global_idx -> (D,) tensor on CPU, indexed in the order
    produced by form_doc_batches(..., shuffle=False).
    """
    batches = form_doc_batches(doc_cls, doc_rates, cls_budget, shuffle=False)
    references = {}
    global_idx = 0

    for cls_tokens, attention_mask, _ in tqdm(batches, desc="  Reference outputs"):
        cls_tokens = cls_tokens.to(device)
        attention_mask = attention_mask.to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = predictor(cls_tokens, attention_mask=attention_mask)
        doc_emb = out["doc_embedding"].float().cpu()

        for i in range(doc_emb.shape[0]):
            references[global_idx] = doc_emb[i]
            global_idx += 1

    return references


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = min(1.0, float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        ))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(path, predictor, optimizer, scheduler, epoch, global_step,
                    ma=None, val_loss=None, args=None, weights_only=False):
    os.makedirs(path, exist_ok=True)

    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": predictor.state_dict(),
    }
    if not weights_only:
        state["optimizer_state_dict"] = optimizer.state_dict()
        state["scheduler_state_dict"] = scheduler.state_dict()
    if ma:
        state["train_loss"] = sum(ma["loss"]) / len(ma["loss"]) if ma["loss"] else float("nan")
    if val_loss is not None:
        state["val_loss"] = val_loss
    if args is not None:
        state["args"] = vars(args)
    torch.save(state, os.path.join(path, "full_model.pt"))

    unwrapped = predictor._orig_mod if hasattr(predictor, "_orig_mod") else predictor
    torch.save(unwrapped.aggregator.state_dict(), os.path.join(path, "aggregator.pt"))
    torch.save(unwrapped.head.state_dict(), os.path.join(path, "head.pt"))
    print(f"  Saved checkpoint to {path}")


# ---------------------------------------------------------------------------
# Epoch
# ---------------------------------------------------------------------------

def run_epoch(predictor, idx_to_rate, device, args, split,
              optimizer=None, scheduler=None, epoch=0, global_step=0,
              training=True, reference_outputs=None):
    prefix = "Train" if training else "Val"
    cache_epoch = epoch % NUM_CACHED_EPOCHS

    t_cache = time.time()
    print(f"  {prefix}: Loading CLS cache (cache_epoch={cache_epoch})...")
    cls_cache = load_cls_cache(args.output_dir, split, cache_epoch)
    doc_cls, doc_rates, _ = align_cache_with_rates(cls_cache, idx_to_rate)
    del cls_cache
    print(f"  [TIMING] load+align CLS cache: {time.time() - t_cache:.1f}s")
    print(f"  {len(doc_cls)} rated documents with CLS embeddings")

    t_batch = time.time()
    batches = form_doc_batches(doc_cls, doc_rates, args.cls_budget,
                               shuffle=training)
    del doc_cls, doc_rates
    print(f"  {len(batches)} batches (cls_budget={args.cls_budget})")
    print(f"  [TIMING] form batches: {time.time() - t_batch:.1f}s")

    if training:
        predictor.train()
        unwrapped = predictor._orig_mod if hasattr(predictor, "_orig_mod") else predictor
        if not args.train_aggregator:
            unwrapped.aggregator.eval()
    else:
        predictor.eval()

    MA_WINDOW = 100
    ma = {k: deque(maxlen=MA_WINDOW) for k in ("loss", "mse", "reg", "mae")}
    total = {k: 0.0 for k in ma}
    n_batches = 0
    accum_count = 0
    last_ckpt_time = time.time()
    doc_counter = 0

    prev_grad = torch.is_grad_enabled()
    if not training:
        torch.set_grad_enabled(False)
    if training:
        optimizer.zero_grad()

    try:
        pbar = tqdm(batches, desc=f"  {prefix} epoch {epoch+1}", unit="batch")
        for batch_idx, (cls_tokens, attention_mask, targets) in enumerate(pbar):
            cls_tokens = cls_tokens.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = predictor(cls_tokens, attention_mask=attention_mask)
                prediction = out["prediction"].float()

            unwrapped = predictor._orig_mod if hasattr(predictor, "_orig_mod") else predictor
            mse_loss = unwrapped.compute_loss(prediction, targets)

            reg_loss = torch.tensor(0.0, device=device)
            if training and args.regularization and reference_outputs is not None:
                doc_emb = out["doc_embedding"].float()
                B = doc_emb.shape[0]
                ref_batch = torch.stack([
                    reference_outputs[doc_counter + i] for i in range(B)
                ]).to(device)
                reg_loss = nn.functional.mse_loss(doc_emb, ref_batch)

            loss = mse_loss + args.reg_lambda * reg_loss

            if training:
                accum_count += 1
                (loss / args.grad_accum_steps).backward()

                if accum_count % args.grad_accum_steps == 0:
                    nn.utils.clip_grad_norm_(
                        [p for p in predictor.parameters() if p.requires_grad],
                        args.max_grad_norm,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

            with torch.no_grad():
                batch_mae = (prediction - targets).abs().mean().item()

            doc_counter += cls_tokens.shape[0]

            ma["loss"].append(loss.item())
            ma["mse"].append(mse_loss.item())
            ma["reg"].append(reg_loss.item())
            ma["mae"].append(batch_mae)
            for k in total:
                total[k] += ma[k][-1]
            n_batches += 1

            def _ma_avg(d):
                return sum(d) / len(d) if d else float("nan")

            if n_batches % 10 == 0 or batch_idx == len(batches) - 1:
                postfix = dict(
                    loss=f"{_ma_avg(ma['loss']):.4f}",
                    mse=f"{_ma_avg(ma['mse']):.4f}",
                    mae=f"{_ma_avg(ma['mae']):.4f}",
                )
                if args.regularization:
                    postfix["reg"] = f"{_ma_avg(ma['reg']):.4f}"
                if training and scheduler is not None:
                    postfix["lr"] = f"{scheduler.get_last_lr()[0]:.2e}"
                pbar.set_postfix(postfix)

            if (training and not args.no_save and args.checkpoint_minutes > 0
                    and time.time() - last_ckpt_time >= args.checkpoint_minutes * 60):
                ckpt_path = os.path.join(args.output_dir, "checkpoint_latest")
                save_checkpoint(ckpt_path, predictor, optimizer, scheduler,
                                epoch, global_step, ma=ma)
                last_ckpt_time = time.time()

        if training and accum_count % args.grad_accum_steps != 0:
            nn.utils.clip_grad_norm_(
                [p for p in predictor.parameters() if p.requires_grad],
                args.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

    finally:
        torch.set_grad_enabled(prev_grad)

    n = max(n_batches, 1)
    metrics = {k: total[k] / n for k in total}
    return metrics, global_step


# ---------------------------------------------------------------------------
# Steps estimation from cache
# ---------------------------------------------------------------------------

def estimate_steps_from_cache(cls_cache, idx_to_rate, cls_budget, grad_accum_steps):
    sizes = sorted(
        cls_cache[k].shape[0] for k in cls_cache if k in idx_to_rate
    )
    n_batches = 0
    current_total = 0
    started = False
    for n in sizes:
        if started and current_total + n > cls_budget:
            n_batches += 1
            current_total = n
        else:
            current_total += n
            started = True
    if started:
        n_batches += 1
    steps = max(1, n_batches // grad_accum_steps)
    print(f"  From cache: {len(sizes)} docs -> {n_batches} batches, "
          f"~{steps} optimizer steps/epoch")
    return steps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("medium")
    print(f"Device: {device}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------
    # Documents + growth rates -> per-split {doc_idx -> rate}
    # -----------------------------------------------------------
    t_section = time.time()
    print(f"\nLoading documents from {args.data}...")
    documents = torch.load(args.data, map_location="cpu", weights_only=False)
    print(f"  [TIMING] torch.load documents: {time.time() - t_section:.1f}s")

    with open(args.growth_rates, "r") as f:
        growth_rates = json.load(f)
    print(f"  {len(growth_rates)} companies with growth rates")

    # Must reproduce the exact same split used by cls_aggregator_training_pipeline
    train_docs, val_docs = split_documents(documents, args.val_ratio)
    del documents
    print(f"  Split: {len(train_docs)} train docs, {len(val_docs)} val docs")

    train_idx_to_rate = build_idx_to_rate(train_docs, growth_rates)
    val_idx_to_rate = build_idx_to_rate(val_docs, growth_rates)
    del growth_rates, train_docs, val_docs
    print(f"  Rated: {len(train_idx_to_rate)} train, {len(val_idx_to_rate)} val")

    if not train_idx_to_rate:
        print("ERROR: No training documents matched to growth rates.")
        return

    # -----------------------------------------------------------
    # Predictor
    # -----------------------------------------------------------
    t_section = time.time()
    print("\nBuilding GrowthPredictor...")
    predictor = build_predictor(
        aggregator_checkpoint=args.aggregator_checkpoint,
        hidden_size=args.hidden_size,
        train_aggregator=args.train_aggregator,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_mult=args.ffn_mult,
    )
    predictor = predictor.to(device)
    if args.compile:
        print("Compiling predictor with torch.compile...")
        predictor = torch.compile(predictor, dynamic=True)
    print(f"  [TIMING] build predictor: {time.time() - t_section:.1f}s")

    # -----------------------------------------------------------
    # Optimizer + schedule
    # -----------------------------------------------------------
    unwrapped = predictor._orig_mod if hasattr(predictor, "_orig_mod") else predictor
    param_groups = [{
        "params": list(unwrapped.head.parameters()),
        "lr": args.lr,
        "name": "head",
    }]
    if args.train_aggregator:
        param_groups.append({
            "params": list(unwrapped.aggregator.parameters()),
            "lr": args.aggregator_lr,
            "name": "aggregator",
        })

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=args.weight_decay,
    )

    # Estimate steps/epoch from the first cached file
    print("\nEstimating steps per epoch...")
    first_cache = load_cls_cache(args.output_dir, "train", 0)
    steps_per_epoch = estimate_steps_from_cache(
        first_cache, train_idx_to_rate, args.cls_budget, args.grad_accum_steps,
    )
    del first_cache

    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # -----------------------------------------------------------
    # Resume
    # -----------------------------------------------------------
    start_epoch = 0
    global_step = 0
    if args.resume_from:
        ckpt_file = os.path.join(args.resume_from, "full_model.pt")
        print(f"\nResuming from {ckpt_file}...")
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
        state_dict = {k.removeprefix("_orig_mod."): v
                      for k, v in ckpt["model_state_dict"].items()}
        unwrapped.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        print(f"  Resumed at epoch={start_epoch}, global_step={global_step}")

    # -----------------------------------------------------------
    # Regularization reference (uses epoch 0 train cache)
    # -----------------------------------------------------------
    reference_outputs = None
    if args.regularization and args.train_aggregator:
        print("\nPre-computing reference aggregator outputs for regularization...")
        ref_cache = load_cls_cache(args.output_dir, "train", 0)
        ref_doc_cls, ref_doc_rates, _ = align_cache_with_rates(
            ref_cache, train_idx_to_rate)
        del ref_cache
        reference_outputs = compute_reference_outputs(
            unwrapped, ref_doc_cls, ref_doc_rates, args.cls_budget, device,
        )
        del ref_doc_cls, ref_doc_rates
        print(f"  Computed {len(reference_outputs)} reference embeddings")

    # -----------------------------------------------------------
    # Config print
    # -----------------------------------------------------------
    print(f"\nTraining: {args.epochs} epochs, ~{steps_per_epoch} steps/epoch, "
          f"~{total_steps} total steps")
    print(f"cls_budget: {args.cls_budget}, grad_accum: {args.grad_accum_steps}")
    print(f"Head LR: {args.lr}, warmup: {warmup_steps} steps")
    if args.train_aggregator:
        print(f"Aggregator LR: {args.aggregator_lr}")
    if args.regularization:
        print(f"Regularization: lambda={args.reg_lambda}")
    print()

    # -----------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------
    best_val_loss = float("inf")
    best_val_mae = float("inf")

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*70}")

        t0 = time.time()
        train_metrics, global_step = run_epoch(
            predictor, train_idx_to_rate, device, args, split="train",
            optimizer=optimizer, scheduler=scheduler,
            epoch=epoch, global_step=global_step,
            training=True, reference_outputs=reference_outputs,
        )
        t_train = time.time() - t0
        print(f"\n  Train: loss={train_metrics['loss']:.4f} "
              f"(mse={train_metrics['mse']:.4f}, "
              f"reg={train_metrics['reg']:.4f}, "
              f"mae={train_metrics['mae']:.4f}) [{t_train:.0f}s]")

        val_loss = None
        if val_idx_to_rate:
            t0 = time.time()
            val_metrics, _ = run_epoch(
                predictor, val_idx_to_rate, device, args, split="val",
                epoch=epoch, training=False,
            )
            t_val = time.time() - t0
            val_loss = val_metrics["loss"]
            print(f"  Val:   loss={val_metrics['loss']:.4f} "
                  f"(mse={val_metrics['mse']:.4f}, "
                  f"mae={val_metrics['mae']:.4f}) [{t_val:.0f}s]")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_mae = val_metrics["mae"]

        if not args.no_save:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch + 1}")
            save_checkpoint(ckpt_path, predictor, optimizer, scheduler,
                            epoch + 1, global_step,
                            val_loss=val_loss, args=args, weights_only=True)
            latest_path = os.path.join(args.output_dir, "checkpoint_latest")
            save_checkpoint(latest_path, predictor, optimizer, scheduler,
                            epoch + 1, global_step,
                            val_loss=val_loss, args=args, weights_only=False)

    if best_val_loss < float("inf"):
        print(f"\nBest val loss: {best_val_loss:.4f} (MAE={best_val_mae:.4f})")
    print("\nTraining complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Growth rate prediction training (uses pre-computed CLS cache)")

    # Data
    parser.add_argument("--data", required=True,
                        help="Path to documents.pt (used only to align cache doc_idx with growth rates)")
    parser.add_argument("--growth-rates", required=True,
                        help="Path to growth_rates.json ({company_id: rate})")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Must match the val_ratio used by cls_aggregator_training_pipeline")

    # Aggregator
    parser.add_argument("--aggregator-checkpoint", type=str, default=None,
                        help="Path to aggregator checkpoint (JEPA-trained)")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--ffn-mult", type=int, default=4)

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--cls-budget", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--aggregator-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    # Aggregator training
    parser.add_argument("--train-aggregator", action="store_true")
    parser.add_argument("--regularization", action="store_true")
    parser.add_argument("--reg-lambda", type=float, default=0.1)

    # Output / resuming
    parser.add_argument("--output-dir", default="checkpoints/growth_predictor",
                        help="Directory for checkpoints and local cache mirror")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--checkpoint-minutes", type=int, default=30)
    parser.add_argument("--no-save", action="store_true")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
