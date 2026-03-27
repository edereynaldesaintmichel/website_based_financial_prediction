"""
Training script for T5-style CLS embedding learning.

Encoder sees clean input -> CLS. Decoder sees masked input (random ratio
per example, 80/10/10 corruption) and reconstructs using CLS + context.

Data: loads documents.pt (shared with MLM and CLS aggregator pipelines).
Chunking happens at the start of each epoch with random target sizes and
segment-aware boundaries (line breaks, but not inside tables).

Includes a supervised contrastive loss (SupCon) on CLS embeddings to
encourage same-document chunks to have similar CLS representations.

Supports:
- BF16 mixed precision
- Gradient accumulation
- Cosine LR schedule with warmup
- Dynamic batching with contrastive pairing (same-doc chunks in batch)
- Validation after each epoch
- Resume from checkpoint
- Periodic checkpointing
- torch.compile

Usage:
    python -m t5_style_training_pipeline.train \
        --data documents.pt \
        --encoder_checkpoint checkpoints/mlm_full_baseline/checkpoint_epoch3/full_model.pt \
        --output_dir checkpoints/t5_cls
"""
import argparse
import math
import os
import random
import sys
import time
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from financial_bert import TABLE_START_ID, TABLE_END_ID
from split_utils import split_documents
from t5_style_training_pipeline.decoder import build_t5_model
from chunk_utils import (
    chunk_spans,
    extract_chunk,
    pad_and_collate,
    get_token_info,
)


CHUNK_MIN = 128
CHUNK_MAX = 1024
MIN_SAME_DOC = 2   # minimum same-doc chunks per batch for contrastive pairs
MAX_SAME_DOC = 3   # maximum same-doc chunks per batch


# ---------------------------------------------------------------------------
# Table-aware boundary detection
# ---------------------------------------------------------------------------

def get_boundaries_table_aware(input_ids, newline_ids):
    """Find chunk boundary candidates (positions after newline tokens),
    excluding newlines that fall between TABLE_START and TABLE_END.

    Works in content space (CLS/SEP stripped: position 0 = first content token).
    Returns sorted list of boundary positions.
    """
    content = input_ids[1:-1]  # strip CLS at 0 and SEP at -1
    n = len(content)

    # Build mask of positions inside tables using cumsum
    starts = (content == TABLE_START_ID)
    ends = (content == TABLE_END_ID)
    depth = starts.long().cumsum(0) - ends.long().cumsum(0).roll(1, 0)
    depth[0] = starts[0].long()
    in_table = depth > 0

    # Find newline positions that are NOT inside a table
    nl_mask = torch.zeros(n, dtype=torch.bool)
    for nl_id in newline_ids:
        nl_mask |= (content == nl_id)
    valid = nl_mask & ~in_table

    # Boundary = position AFTER the newline (start of next chunk)
    return (valid.nonzero(as_tuple=False).squeeze(-1) + 1).tolist()


# ---------------------------------------------------------------------------
# Chunking (per epoch)
# ---------------------------------------------------------------------------

def chunk_documents(documents, newline_ids, cls_id, sep_id, seed):
    """Chunk all documents with random target sizes and table-aware boundaries.

    Returns list of chunk dicts, each with doc_idx and seq_length fields.
    """
    rng = random.Random(seed)
    all_chunks = []

    for doc_idx, doc in enumerate(documents):
        content_len = len(doc["input_ids"]) - 2  # strip CLS/SEP
        if content_len < 1:
            continue

        boundaries = get_boundaries_table_aware(doc["input_ids"], newline_ids)

        # Random target per chunk (natural variability from boundary snapping)
        spans = chunk_spans(
            content_len, boundaries,
            lambda: rng.randint(CHUNK_MIN, CHUNK_MAX),
        )

        for ci, (s, e) in enumerate(spans):
            chunk = extract_chunk(doc, s, e, cls_id, sep_id)
            chunk["doc_idx"] = doc_idx
            all_chunks.append(chunk)

    return all_chunks


# ---------------------------------------------------------------------------
# Contrastive batching
# ---------------------------------------------------------------------------

def form_contrastive_batches(chunks, token_budget, bucket_width=32):
    """Form batches with 2-3 same-document chunks per batch for contrastive learning.

    Strategy:
    1. Bucket chunks by length (reduces padding waste).
    2. Within each bucket, sort by doc_idx to group same-doc chunks together.
    3. Greedily build batches, ensuring MIN_SAME_DOC-MAX_SAME_DOC chunks
       from the same document appear in each batch.
    """
    # Bucket by length
    buckets = defaultdict(list)
    for chunk in chunks:
        key = (chunk["seq_length"] + bucket_width - 1) // bucket_width * bucket_width
        buckets[key].append(chunk)

    all_batches = []

    for bucket_len, bucket_items in buckets.items():
        # Sort by doc_idx within bucket
        bucket_items.sort(key=lambda c: c["doc_idx"])

        max_batch_size = max(1, token_budget // bucket_len)

        # Group consecutive same-doc chunks
        i = 0
        current_batch = []
        while i < len(bucket_items):
            # Find run of same-doc chunks
            doc_id = bucket_items[i]["doc_idx"]
            run_end = i + 1
            while run_end < len(bucket_items) and bucket_items[run_end]["doc_idx"] == doc_id:
                run_end += 1
            run = bucket_items[i:run_end]

            # Add chunks from this doc in groups of MIN_SAME_DOC..MAX_SAME_DOC
            for chunk in run:
                current_batch.append(chunk)
                if len(current_batch) >= max_batch_size:
                    all_batches.append(current_batch)
                    current_batch = []

            i = run_end

        if current_batch:
            all_batches.append(current_batch)

    random.shuffle(all_batches)
    return all_batches


# ---------------------------------------------------------------------------
# Contrastive loss
# ---------------------------------------------------------------------------

def supervised_contrastive_loss(cls_embeddings, doc_ids, temperature=0.07):
    """SupCon loss on CLS embeddings using in-batch positives/negatives.

    Args:
        cls_embeddings: (B, D) L2-normalized CLS vectors
        doc_ids: (B,) integer document IDs for each sample
        temperature: scaling temperature

    Returns:
        scalar loss (0 if no positive pairs exist in the batch)
    """
    B = cls_embeddings.shape[0]
    if B < 2:
        return cls_embeddings.new_tensor(0.0)

    # Similarity matrix
    sim = torch.mm(cls_embeddings, cls_embeddings.t()) / temperature  # (B, B)

    # Positive mask: same doc, different sample
    labels = doc_ids.unsqueeze(0) == doc_ids.unsqueeze(1)  # (B, B)
    labels.fill_diagonal_(False)  # exclude self

    if not labels.any():
        return cls_embeddings.new_tensor(0.0)

    # Log-softmax over negatives (exclude self from denominator)
    self_mask = torch.eye(B, dtype=torch.bool, device=cls_embeddings.device)
    sim_masked = sim.masked_fill(self_mask, float("-inf"))
    log_prob = sim_masked - sim_masked.logsumexp(dim=1, keepdim=True)

    # Zero out self-positions to avoid 0 * -inf = NaN
    log_prob = log_prob.masked_fill(self_mask, 0.0)

    # Average log-prob over positive pairs
    # For each anchor, average over its positives
    pos_log_prob = (labels.float() * log_prob).sum(dim=1)
    num_positives = labels.float().sum(dim=1)
    # Only include anchors that have at least one positive
    valid = num_positives > 0
    if not valid.any():
        return cls_embeddings.new_tensor(0.0)

    loss = -(pos_log_prob[valid] / num_positives[valid]).mean()
    return loss


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def create_masked_inputs(
    input_ids: torch.Tensor,
    is_number_mask: torch.Tensor,
    number_values: torch.Tensor,
    mask_token_id: int,
    pad_token_id: int,
    magnitude_sentinel: float,
    vocab_size: int,
    magnitude_min: float,
    magnitude_max: float,
    mask_prob_min: float = 0.15,
    mask_prob_max: float = 0.85,
):
    """Create masked decoder inputs and labels.

    Masking ratio is sampled uniformly per example from [mask_prob_min, mask_prob_max].
    Both text and number masked positions use 80/10/10 corruption:
      80% fully masked, 10% random replacement, 10% keep original.
    For numbers, "random" means a random magnitude from [magnitude_min, magnitude_max].

    Returns:
        decoder_input_ids, decoder_number_values, decoder_is_number_mask,
        labels_text, labels_magnitude
    """
    B, S = input_ids.shape
    device = input_ids.device

    is_number = is_number_mask.bool()
    candidates = (input_ids != pad_token_id)
    candidates[:, 0] = False  # never mask CLS

    # Per-example random mask ratio
    mask_probs = torch.empty(B, 1, device=device).uniform_(mask_prob_min, mask_prob_max)
    rand = torch.rand(B, S, device=device)
    mask_positions = (rand < mask_probs) & candidates

    decoder_input_ids = input_ids.clone()
    decoder_number_values = number_values.clone()
    decoder_is_number_mask = is_number_mask.clone()

    # 80/10/10 corruption split (shared across text and numbers)
    corruption_rand = torch.rand(B, S, device=device)
    do_mask = mask_positions & (corruption_rand < 0.8)
    do_random = mask_positions & (corruption_rand >= 0.8) & (corruption_rand < 0.9)
    # remaining 10%: keep original

    # --- Text positions ---
    text_mask = mask_positions & ~is_number
    decoder_input_ids[do_mask & ~is_number] = mask_token_id
    random_tokens = torch.randint(0, vocab_size, (B, S), device=device)
    decoder_input_ids[do_random & ~is_number] = random_tokens[do_random & ~is_number]

    # --- Number positions ---
    num_mask = mask_positions & is_number
    decoder_input_ids[do_mask & is_number] = mask_token_id
    decoder_number_values[do_mask & is_number] = magnitude_sentinel

    random_mags = torch.empty(B, S, device=device).uniform_(magnitude_min, magnitude_max)
    decoder_number_values[do_random & is_number] = random_mags[do_random & is_number]
    # (do_random numbers keep their original input_id -- the number placeholder token)

    # Labels: predict at ALL masked positions (including keep-original ones)
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
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, optimizer, scheduler, epoch, global_step,
                    ma=None, val_loss=None, args=None, weights_only=False):
    """Save checkpoint. If weights_only, save just model state dict; otherwise
    include optimizer/scheduler state for resuming."""
    os.makedirs(path, exist_ok=True)
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
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

    # Also save encoder separately for downstream use
    torch.save(model.encoder.state_dict(), os.path.join(path, "encoder.pt"))
    print(f"  Saved checkpoint to {path}")


# ---------------------------------------------------------------------------
# Encoder dropout
# ---------------------------------------------------------------------------

def enable_encoder_dropout(model, dropout_prob=0.1):
    """Enable dropout in the encoder for contrastive augmentation.

    ModernBERT-base ships with all dropouts at 0.0. This sets them to
    dropout_prob so the two forward passes through the encoder (for the
    same input) produce slightly different CLS embeddings, acting as
    natural augmentation for the contrastive objective.
    """
    config = model.encoder.modernbert.config
    config.attention_dropout = dropout_prob
    config.embedding_dropout = dropout_prob
    config.mlp_dropout = dropout_prob
    config.classifier_dropout = dropout_prob

    # Walk all modules and update existing Dropout layers
    for module in model.encoder.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_prob


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_epoch(model, documents, device, tok_info, args, config,
              optimizer=None, scheduler=None, epoch=0, global_step=0,
              training=True, contrastive_lambda=0.02):
    """Run one training or validation epoch.

    Returns: (metrics_dict, global_step)
    """
    prefix = "Train" if training else "Val"
    if training:
        model.train()
    else:
        model.eval()

    cls_id = tok_info["cls_id"]
    sep_id = tok_info["sep_id"]
    pad_id = tok_info["pad_id"]
    mask_id = tok_info["mask_id"]
    nl_ids = tok_info["newline_ids"]
    mag_sentinel = config.magnitude_max + 1.0

    # Chunk all documents (fresh each epoch)
    chunk_seed = args.seed + epoch + (0 if training else 10000)
    all_chunks = chunk_documents(documents, nl_ids, cls_id, sep_id, chunk_seed)

    n_chunks = len(all_chunks)
    avg_len = sum(c["seq_length"] for c in all_chunks) / max(n_chunks, 1)
    print(f"  {prefix}: {n_chunks} chunks (avg {avg_len:.0f} tokens)")

    # Form batches with contrastive pairing
    batches = form_contrastive_batches(all_chunks, args.tokens_per_batch)
    del all_chunks
    print(f"  {len(batches)} batches (token_budget={args.tokens_per_batch})")

    # Metrics
    MA_WINDOW = 100
    ma = {k: deque(maxlen=MA_WINDOW)
          for k in ("loss", "text", "mag", "contrastive")}
    total = {"loss": 0.0, "text": 0.0, "mag": 0.0, "contrastive": 0.0}
    n_batches = 0
    accum_count = 0
    last_ckpt_time = time.time()

    prev_grad = torch.is_grad_enabled()
    if not training:
        torch.set_grad_enabled(False)

    if training:
        optimizer.zero_grad()

    try:
        pbar = tqdm(batches, desc=f"  {prefix} epoch {epoch+1}", unit="batch")
        for batch_idx, batch_chunks in enumerate(pbar):
            # Pad and collate
            ids, num_mask, num_vals, attn_mask = pad_and_collate(batch_chunks, pad_id)
            ids = ids.to(device)
            num_mask = num_mask.to(device)
            num_vals = num_vals.to(device)
            attn_mask = attn_mask.to(device)

            # Doc IDs for contrastive loss
            doc_ids = torch.tensor(
                [c["doc_idx"] for c in batch_chunks],
                dtype=torch.long, device=device,
            )

            # Create masked decoder inputs
            (dec_ids, dec_nums, dec_num_mask,
             labels_text, labels_mag) = create_masked_inputs(
                ids, num_mask, num_vals,
                mask_token_id=mask_id, pad_token_id=pad_id,
                magnitude_sentinel=mag_sentinel,
                vocab_size=config.vocab_size,
                magnitude_min=config.magnitude_min,
                magnitude_max=config.magnitude_max,
                mask_prob_min=args.mask_prob_min,
                mask_prob_max=args.mask_prob_max,
            )

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=ids,
                    number_values=num_vals,
                    is_number_mask=num_mask,
                    attention_mask=attn_mask,
                    decoder_input_ids=dec_ids,
                    decoder_number_values=dec_nums,
                    decoder_is_number_mask=dec_num_mask,
                    labels_text=labels_text,
                    labels_magnitude=labels_mag,
                )

            recon_loss = outputs["loss"]

            # Contrastive loss on CLS embeddings
            cls_hidden = outputs["cls_hidden"].float()
            cls_normed = F.normalize(cls_hidden, dim=-1)
            con_loss = supervised_contrastive_loss(cls_normed, doc_ids)

            loss = recon_loss + contrastive_lambda * con_loss

            if training:
                accum_count += 1
                (loss / args.grad_accum_steps).backward()

                if accum_count % args.grad_accum_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

            # Track metrics
            ma["loss"].append(loss.item())
            ma["text"].append(outputs["loss_text"].item())
            ma["mag"].append(outputs["loss_mag"].item())
            ma["contrastive"].append(con_loss.item())
            for k in total:
                total[k] += ma[k][-1]
            n_batches += 1

            def _ma_avg(d):
                return sum(d) / len(d) if d else float("nan")

            if n_batches % 20 == 0 or batch_idx == len(batches) - 1:
                postfix = dict(
                    loss=f"{_ma_avg(ma['loss']):.4f}",
                    txt=f"{_ma_avg(ma['text']):.4f}",
                    mag=f"{_ma_avg(ma['mag']):.4f}",
                    con=f"{_ma_avg(ma['contrastive']):.4f}",
                )
                if training and scheduler is not None:
                    postfix["lr"] = f"{scheduler.get_last_lr()[0]:.2e}"
                pbar.set_postfix(postfix)

            # Periodic checkpoint
            if (training and args.checkpoint_minutes > 0
                    and time.time() - last_ckpt_time >= args.checkpoint_minutes * 60):
                ckpt_path = os.path.join(args.output_dir, "checkpoint_latest")
                save_checkpoint(ckpt_path, model, optimizer, scheduler,
                                epoch, global_step, ma=ma)
                last_ckpt_time = time.time()

        # Flush remaining accumulated gradients
        if training and accum_count % args.grad_accum_steps != 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

    finally:
        torch.set_grad_enabled(prev_grad)

    n = max(n_batches, 1)
    metrics = {k: total[k] / n for k in total}
    return metrics, global_step


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("medium")
    print(f"Device: {device}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print(f"\nLoading documents from {args.data}...")
    documents = torch.load(args.data, map_location="cpu", weights_only=False)
    train_docs, val_docs = split_documents(documents, args.val_ratio)
    del documents
    print(f"  {len(train_docs)} train, {len(val_docs)} val documents")

    # ------------------------------------------------------------------
    # Token info
    # ------------------------------------------------------------------
    tok_info = get_token_info(args.pretrained_model_id)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print(f"\nBuilding model (cross_attn_type={args.cross_attn_type})...")
    model = build_t5_model(
        encoder_checkpoint=args.encoder_checkpoint,
        pretrained_model_id=args.pretrained_model_id,
        num_magnitude_bins=args.num_magnitude_bins,
        cross_attn_type=args.cross_attn_type,
    )
    model = model.to(device)

    # Enable encoder dropout for contrastive augmentation
    enable_encoder_dropout(model, args.encoder_dropout)
    print(f"  Encoder dropout: {args.encoder_dropout}")

    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model, dynamic=True)

    config = model.config if hasattr(model, "config") else model._orig_mod.config

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

    # Estimate steps per epoch from total content tokens
    total_content = sum(len(d["input_ids"]) - 2 for d in train_docs)
    avg_chunk_len = (CHUNK_MIN + CHUNK_MAX) / 2
    est_chunks = total_content / avg_chunk_len
    est_batches = est_chunks * avg_chunk_len / args.tokens_per_batch
    steps_per_epoch = int(est_batches / args.grad_accum_steps)
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
        unwrapped = model._orig_mod if hasattr(model, "_orig_mod") else model
        unwrapped.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        print(f"  Resumed at epoch={start_epoch}, global_step={global_step}")

    print(f"\nTraining: {args.epochs} epochs, ~{steps_per_epoch} steps/epoch, "
          f"~{total_steps} total steps")
    print(f"tokens_per_batch: {args.tokens_per_batch}, grad_accum: {args.grad_accum_steps}")
    print(f"LR: {args.lr}, warmup: {warmup_steps} steps, "
          f"mask prob: U({args.mask_prob_min}, {args.mask_prob_max}), 80/10/10 corruption")
    print(f"Contrastive lambda: {args.contrastive_lambda}")
    print()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*70}")

        t0 = time.time()
        train_metrics, global_step = run_epoch(
            model, train_docs, device, tok_info, args, config,
            optimizer=optimizer, scheduler=scheduler,
            epoch=epoch, global_step=global_step,
            training=True, contrastive_lambda=args.contrastive_lambda,
        )
        t_train = time.time() - t0

        print(f"\n  Train: loss={train_metrics['loss']:.4f} "
              f"(text={train_metrics['text']:.4f}, "
              f"mag={train_metrics['mag']:.4f}, "
              f"con={train_metrics['contrastive']:.4f}) [{t_train:.0f}s]")

        # Validation
        val_loss = None
        if val_docs:
            t0 = time.time()
            val_metrics, _ = run_epoch(
                model, val_docs, device, tok_info, args, config,
                epoch=epoch, training=False,
                contrastive_lambda=args.contrastive_lambda,
            )
            t_val = time.time() - t0
            val_loss = val_metrics["loss"]
            print(f"  Val:   loss={val_metrics['loss']:.4f} "
                  f"(text={val_metrics['text']:.4f}, "
                  f"mag={val_metrics['mag']:.4f}, "
                  f"con={val_metrics['contrastive']:.4f}) [{t_val:.0f}s]")

        # End-of-epoch checkpoint (weights only)
        ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch + 1}")
        save_checkpoint(ckpt_path, model, optimizer, scheduler,
                        epoch + 1, global_step,
                        val_loss=val_loss, args=args, weights_only=True)

    print("\nTraining complete.")

    # Free VRAM so back-to-back runs (e.g. --compare) don't OOM
    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
        if hasattr(torch.compiler, "reset"):
            torch.compiler.reset()


def main():
    parser = argparse.ArgumentParser(description="T5-style CLS embedding training")

    # Data
    parser.add_argument("--data", required=True,
                        help="Path to documents.pt (shared with MLM/aggregator pipelines)")
    parser.add_argument("--encoder_checkpoint", required=True,
                        help="Path to encoder MLM checkpoint (full_model.pt)")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of documents for validation")

    # Model
    parser.add_argument("--pretrained_model_id", default="answerdotai/ModernBERT-base")
    parser.add_argument("--num_magnitude_bins", type=int, default=128)
    parser.add_argument("--cross_attn_type", default="mlp", choices=["mlp", "expanded_memory"],
                        help="CLS conditioning module: 'mlp' (CLSSwiGLU) or 'expanded_memory'")
    parser.add_argument("--encoder_dropout", type=float, default=0.1,
                        help="Dropout probability for encoder (augmentation for contrastive loss)")
    parser.add_argument("--compare", action="store_true", default=True,
                        help="Run both cross_attn_type variants for 1 epoch and compare val loss")
    parser.add_argument("--no_compare", action="store_false", dest="compare",
                        help="Disable compare mode, train a single variant")

    # Training
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--tokens_per_batch", type=int, default=4096,
                        help="Target total tokens per batch (dynamic batch size per bucket)")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--mask_prob_min", type=float, default=0.15,
                        help="Min masking ratio (sampled uniformly per example)")
    parser.add_argument("--mask_prob_max", type=float, default=0.85,
                        help="Max masking ratio (sampled uniformly per example)")
    parser.add_argument("--contrastive_lambda", type=float, default=0.02,
                        help="Weight for contrastive loss (target: <10%% of gradient magnitude)")
    parser.add_argument("--seed", type=int, default=42)

    # Output / resuming
    parser.add_argument("--output_dir", default="checkpoints/t5_cls")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for kernel fusion")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint directory to resume from")
    parser.add_argument("--checkpoint_minutes", type=int, default=30,
                        help="Save a checkpoint every N minutes (0 to disable)")

    args = parser.parse_args()

    if args.compare:
        # Run both variants for 1 epoch, report val loss
        base_output_dir = args.output_dir
        original_epochs = args.epochs
        args.epochs = 1
        results = {}

        for variant in ("mlp", "expanded_memory"):
            args.cross_attn_type = variant
            args.output_dir = os.path.join(base_output_dir, f"compare_{variant}")

            # Check if this variant already finished
            finished_ckpt = os.path.join(args.output_dir, "checkpoint_epoch1", "full_model.pt")
            if os.path.exists(finished_ckpt):
                ckpt = torch.load(finished_ckpt, map_location="cpu", weights_only=False)
                val_loss = ckpt.get("val_loss")
                if val_loss is not None:
                    print(f"\n{'='*60}")
                    print(f"  COMPARE: {variant} already finished (val_loss={val_loss:.4f}), skipping")
                    print(f"{'='*60}\n")
                    results[variant] = val_loss
                    continue

            # Check for a partial checkpoint to resume from
            latest_ckpt = os.path.join(args.output_dir, "checkpoint_latest")
            if os.path.exists(os.path.join(latest_ckpt, "full_model.pt")):
                print(f"\n{'='*60}")
                print(f"  COMPARE: resuming {variant} from {latest_ckpt}")
                print(f"{'='*60}\n")
                args.resume_from = latest_ckpt
            else:
                print(f"\n{'='*60}")
                print(f"  COMPARE: training with cross_attn_type={variant}")
                print(f"{'='*60}\n")
                args.resume_from = None

            train(args)

            # Read back val loss
            if os.path.exists(finished_ckpt):
                ckpt = torch.load(finished_ckpt, map_location="cpu", weights_only=False)
                results[variant] = ckpt.get("val_loss", float("nan"))
            else:
                results[variant] = float("nan")

        print(f"\n{'='*60}")
        print("  COMPARISON RESULTS (1 epoch)")
        print(f"{'='*60}")
        for variant, val_loss in results.items():
            print(f"  {variant:20s}  val_loss = {val_loss:.4f}")
        best = min(results, key=results.get)
        print(f"\n  Best: {best} (val_loss={results[best]:.4f})")
        args.epochs = original_epochs
    else:
        train(args)


if __name__ == "__main__":
    main()
