"""
Training script for growth rate prediction from document embeddings.

Pipeline:
1. Load documents + growth rates, match by company ID
2. Each epoch: chunk documents, compute CLS embeddings (frozen encoder),
   group by document, forward through GrowthPredictor (aggregator + head)
3. Train with smoothed cross-entropy + optional aggregator regularization

Supports:
- BF16 mixed precision
- Gradient accumulation
- Cosine LR schedule with warmup
- Frozen or trainable aggregator (separate LR)
- Aggregator output regularization
- Resume from checkpoint
- Periodic checkpointing
- torch.compile

Usage:
    python -m final_training_pipeline.train \
        --data mlm_data/documents.pt \
        --growth-rates growth_rates.json \
        --encoder-checkpoint checkpoints/t5_cls/checkpoint_epoch5/full_model.pt \
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

from chunk_utils import (
    get_token_info,
    get_boundaries,
    chunk_spans,
    extract_chunk,
    pad_and_collate,
)
from financial_bert import FinancialModernBertConfig
from split_utils import split_documents
from t5_style_training_pipeline.decoder import T5StyleModel
from final_training_pipeline.model import GrowthPredictor
from cls_aggregator_training_pipeline.aggregator import CLSAggregator


PRETRAINED_ID = "answerdotai/ModernBERT-base"
CHUNK_MIN = 128
CHUNK_MAX = 1024


# ---------------------------------------------------------------------------
# Company ID extraction
# ---------------------------------------------------------------------------

def extract_company_id(source_file):
    """Extract company identifier from source_file name.

    SEC:       '1000045_2018-06-27.md' -> '1000045'
    UK:        'Prod224_0052_00781277_20171231.md' -> '00781277'
    Wikipedia: '000000_alabama.txt' -> None (no growth data)
    """
    if source_file.endswith('.txt'):
        return None  # Wikipedia
    if source_file.startswith('Prod'):
        parts = source_file.split('_')
        if len(parts) >= 4:
            return parts[2]  # company_id
    else:
        # SEC: CIK_DATE.md
        return source_file.split('_')[0]
    return None


# ---------------------------------------------------------------------------
# Data matching
# ---------------------------------------------------------------------------

def match_documents_to_growth_rates(documents, growth_rates):
    """Match documents to growth rates by company ID.

    Returns list of (doc, growth_rate) pairs for documents with a match.
    """
    matched = []
    skipped_wiki = 0
    skipped_no_rate = 0

    for doc in documents:
        cid = extract_company_id(doc["source_file"])
        if cid is None:
            skipped_wiki += 1
            continue
        if cid not in growth_rates:
            skipped_no_rate += 1
            continue
        matched.append((doc, growth_rates[cid]))

    print(f"  Matched: {len(matched)} documents")
    print(f"  Skipped: {skipped_wiki} Wikipedia, {skipped_no_rate} no growth rate")
    return matched


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_encoder(checkpoint_path, device):
    """Load T5 model with frozen encoder."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = FinancialModernBertConfig.from_pretrained(PRETRAINED_ID)
    config.num_magnitude_bins = ckpt["args"].get("num_magnitude_bins", 128)

    model = T5StyleModel(config)
    state_dict = {k.removeprefix("_orig_mod."): v
                  for k, v in ckpt["model_state_dict"].items()}
    del ckpt
    model.load_state_dict(state_dict)
    del state_dict

    model.encoder.eval()
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    model.to(device)

    n_enc = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    print(f"  Encoder: {n_enc:.1f}M parameters (frozen)")
    return model, config


def build_predictor(aggregator_checkpoint, hidden_size, n_bins, min_val, max_val,
                    train_aggregator):
    """Build GrowthPredictor and load aggregator weights."""
    aggregator = CLSAggregator(hidden_size=hidden_size)

    if aggregator_checkpoint:
        ckpt = torch.load(aggregator_checkpoint, map_location="cpu",
                          weights_only=False)
        # Full training checkpoint: extract aggregator sub-dict
        if isinstance(ckpt, dict) and "aggregator" in ckpt:
            state_dict = ckpt["aggregator"]
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
        # Strip prefixes if present
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

    predictor = GrowthPredictor(
        aggregator=aggregator,
        hidden_size=hidden_size,
        n_bins=n_bins,
        min_val=min_val,
        max_val=max_val,
    )
    n_head = sum(p.numel() for p in predictor.head.parameters()) / 1e6
    n_agg = sum(p.numel() for p in predictor.aggregator.parameters()) / 1e6
    print(f"  GrowthPredictor: aggregator {n_agg:.1f}M, head {n_head:.2f}M")
    return predictor


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_documents(docs_with_rates, nl_ids, cls_id, sep_id, seed):
    """Chunk all documents with consistent random target size per doc.

    Returns list of chunk dicts with doc_idx, chunk_idx, and growth_rate fields.
    """
    rng = random.Random(seed)
    all_chunks = []

    for doc_idx, (doc, rate) in enumerate(docs_with_rates):
        content_len = len(doc["input_ids"]) - 2  # strip CLS/SEP
        if content_len < 1:
            continue

        boundaries = get_boundaries(doc["input_ids"], nl_ids)
        enc_target = rng.randint(CHUNK_MIN, CHUNK_MAX)  # ONE target per doc
        spans = chunk_spans(content_len, boundaries, lambda: enc_target)

        for ci, (s, e) in enumerate(spans):
            chunk = extract_chunk(doc, s, e, cls_id, sep_id)
            chunk["doc_idx"] = doc_idx
            chunk["chunk_idx"] = ci
            chunk["growth_rate"] = rate
            all_chunks.append(chunk)

    return all_chunks


# ---------------------------------------------------------------------------
# CLS computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_all_cls(encoder_model, chunks, device, pad_id, token_budget):
    """Compute CLS embeddings for all chunks using the frozen encoder.

    Returns dict: doc_idx -> {chunk_idx: cls_vec (D,) on CPU}.
    """
    # Sort by length for efficient batching
    sorted_chunks = sorted(chunks, key=lambda c: c["seq_length"])

    # Form batches by token budget
    batches = []
    current = []
    current_max = 0
    for chunk in sorted_chunks:
        l = chunk["seq_length"]
        new_max = max(current_max, l)
        if current and new_max * (len(current) + 1) > token_budget:
            batches.append(current)
            current = [chunk]
            current_max = l
        else:
            current.append(chunk)
            current_max = new_max
    if current:
        batches.append(current)

    results = {}  # doc_idx -> {chunk_idx -> cls_vec}

    pbar = tqdm(batches, desc="  CLS embeddings", unit="batch")
    for batch in pbar:
        ids, num_mask, num_vals, attn_mask = pad_and_collate(batch, pad_id)
        ids = ids.to(device)
        num_mask = num_mask.to(device)
        num_vals = num_vals.to(device)
        attn_mask = attn_mask.to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            embeds = encoder_model._build_embeds(
                ids, num_vals, num_mask,
                encoder_model._get_encoder_tok_embeddings(),
                encoder_model.encoder.number_embedder,
            )
            out = encoder_model.encoder.modernbert(
                inputs_embeds=embeds, attention_mask=attn_mask,
            )
        cls_vecs = out.last_hidden_state[:, 0, :].float().cpu()

        for i, chunk in enumerate(batch):
            doc_idx = chunk["doc_idx"]
            chunk_idx = chunk["chunk_idx"]
            if doc_idx not in results:
                results[doc_idx] = {}
            results[doc_idx][chunk_idx] = cls_vecs[i]

    torch.cuda.empty_cache()
    return results


def gather_doc_embeddings(cls_dict, docs_with_rates):
    """Assemble CLS dicts into per-document tensors and growth rate targets.

    Returns:
        doc_cls:    list of (N_i, D) tensors (one per document)
        doc_rates:  list of float growth rates
    """
    doc_cls = []
    doc_rates = []

    for doc_idx in sorted(cls_dict.keys()):
        chunk_dict = cls_dict[doc_idx]
        # Stack in chunk order
        n_chunks = max(chunk_dict.keys()) + 1
        vecs = [chunk_dict[ci] for ci in range(n_chunks)]
        doc_cls.append(torch.stack(vecs, dim=0))  # (N_i, D)
        doc_rates.append(docs_with_rates[doc_idx][1])

    return doc_cls, doc_rates


# ---------------------------------------------------------------------------
# Document batching
# ---------------------------------------------------------------------------

def form_doc_batches(doc_cls, doc_rates, cls_budget, shuffle=True):
    """Group documents into batches by greedily packing up to cls_budget total CLS tokens.

    Documents are sorted by number of CLS embeddings to reduce padding waste,
    then greedily packed. Resulting batches are shuffled for training.

    Returns list of (cls_tokens, attention_mask, targets) tuples.
    """
    # Sort by n_chunks so similar-length docs are grouped (less padding)
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

        # Pad chunk dimension
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


def estimate_steps_per_epoch(docs_with_rates, tok_info, seed, cls_budget,
                             grad_accum_steps):
    """Dry-run chunking (CPU only) to count batches per epoch."""
    chunks = chunk_documents(docs_with_rates, tok_info["newline_ids"],
                             tok_info["cls_id"], tok_info["sep_id"], seed)

    # Count chunks per document
    doc_n_chunks = {}
    for c in chunks:
        d = c["doc_idx"]
        doc_n_chunks[d] = doc_n_chunks.get(d, 0) + 1

    # Simulate greedy batch packing (sorted by n_chunks)
    sorted_counts = sorted(doc_n_chunks.values())
    n_batches = 0
    current_total = 0
    started = False

    for n in sorted_counts:
        if started and current_total + n > cls_budget:
            n_batches += 1
            current_total = n
        else:
            current_total += n
            started = True
    if started:
        n_batches += 1

    steps = max(1, n_batches // grad_accum_steps)
    n_docs = len(doc_n_chunks)
    n_chunks = len(chunks)
    avg_len = sum(c["seq_length"] for c in chunks) / max(n_chunks, 1)
    print(f"  Dry-run chunking: {n_chunks} chunks from {n_docs} docs "
          f"(avg {avg_len:.0f} tokens) -> {n_batches} batches, "
          f"~{steps} optimizer steps/epoch")
    return steps


# ---------------------------------------------------------------------------
# Regularization: pre-compute reference aggregator outputs
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_reference_outputs(predictor, doc_cls, doc_rates, cls_budget, device):
    """Pre-compute aggregator outputs before training as regularization targets.

    Returns dict: doc_global_idx -> (D,) tensor on CPU.
    """
    batches = form_doc_batches(doc_cls, doc_rates, cls_budget, shuffle=False)
    references = {}
    global_idx = 0

    for cls_tokens, attention_mask, _ in tqdm(batches, desc="  Reference outputs"):
        cls_tokens = cls_tokens.to(device)
        attention_mask = attention_mask.to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            doc_emb = predictor.aggregator(cls_tokens, attention_mask=attention_mask)
        doc_emb = doc_emb.float().cpu()

        for i in range(doc_emb.shape[0]):
            references[global_idx] = doc_emb[i]
            global_idx += 1

    return references


# ---------------------------------------------------------------------------
# LR Schedule
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
    """Save checkpoint."""
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

    # Save aggregator and head separately for downstream use
    unwrapped = predictor._orig_mod if hasattr(predictor, "_orig_mod") else predictor
    torch.save(unwrapped.aggregator.state_dict(), os.path.join(path, "aggregator.pt"))
    torch.save(unwrapped.head.state_dict(), os.path.join(path, "head.pt"))
    print(f"  Saved checkpoint to {path}")


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def run_epoch(predictor, docs_with_rates, encoder_model, device, tok_info,
              args, optimizer=None, scheduler=None, epoch=0, global_step=0,
              training=True, reference_outputs=None):
    """Run one training or validation epoch.

    Returns: (metrics_dict, global_step)
    """
    prefix = "Train" if training else "Val"

    cls_id = tok_info["cls_id"]
    sep_id = tok_info["sep_id"]
    pad_id = tok_info["pad_id"]
    nl_ids = tok_info["newline_ids"]

    # 1. Chunk documents (fresh each epoch)
    chunk_seed = args.seed + epoch + (0 if training else 10000)
    all_chunks = chunk_documents(docs_with_rates, nl_ids, cls_id, sep_id, chunk_seed)

    n_chunks = len(all_chunks)
    avg_len = sum(c["seq_length"] for c in all_chunks) / max(n_chunks, 1)
    print(f"  {prefix}: {n_chunks} chunks from {len(docs_with_rates)} docs "
          f"(avg {avg_len:.0f} tokens)")

    # 2. Compute CLS embeddings (frozen encoder)
    cls_dict = compute_all_cls(encoder_model, all_chunks, device, pad_id,
                               args.token_budget)
    del all_chunks

    # 3. Gather per-document
    doc_cls, doc_rates = gather_doc_embeddings(cls_dict, docs_with_rates)
    del cls_dict
    print(f"  {len(doc_cls)} documents with CLS embeddings")

    # 4. Form document batches
    batches = form_doc_batches(doc_cls, doc_rates, args.cls_budget,
                               shuffle=training)
    del doc_cls, doc_rates
    print(f"  {len(batches)} batches (cls_budget={args.cls_budget})")

    # 5. Forward/backward
    if training:
        predictor.train()
        # Keep aggregator in eval mode if frozen
        unwrapped = predictor._orig_mod if hasattr(predictor, "_orig_mod") else predictor
        if not args.train_aggregator:
            unwrapped.aggregator.eval()
    else:
        predictor.eval()

    MA_WINDOW = 100
    ma = {k: deque(maxlen=MA_WINDOW) for k in ("loss", "ce", "reg", "mae")}
    total = {k: 0.0 for k in ma}
    n_batches = 0
    accum_count = 0
    last_ckpt_time = time.time()
    doc_counter = 0  # global doc index for reference lookup
    lr_exhausted = False

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
                logits = out["logits"].float()

            ce_loss = predictor.compute_loss(logits, targets) if not hasattr(predictor, "_orig_mod") \
                else predictor._orig_mod.compute_loss(logits, targets)

            # Optional aggregator regularization
            reg_loss = torch.tensor(0.0, device=device)
            if training and args.regularization and reference_outputs is not None:
                doc_emb = out["doc_embedding"].float()
                B = doc_emb.shape[0]
                ref_batch = torch.stack([
                    reference_outputs[doc_counter + i] for i in range(B)
                ]).to(device)
                reg_loss = nn.functional.mse_loss(doc_emb, ref_batch)

            loss = ce_loss + args.reg_lambda * reg_loss

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

                    if scheduler.get_last_lr()[0] < 1e-8:
                        lr_exhausted = True

            # Compute MAE for monitoring
            with torch.no_grad():
                preds = predictor.predict(cls_tokens, attention_mask) if not hasattr(predictor, "_orig_mod") \
                    else predictor._orig_mod.predict(cls_tokens, attention_mask)
                batch_mae = (preds - targets).abs().mean().item()

            doc_counter += cls_tokens.shape[0]

            # Track metrics
            ma["loss"].append(loss.item())
            ma["ce"].append(ce_loss.item())
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
                    ce=f"{_ma_avg(ma['ce']):.4f}",
                    mae=f"{_ma_avg(ma['mae']):.4f}",
                )
                if args.regularization:
                    postfix["reg"] = f"{_ma_avg(ma['reg']):.4f}"
                if training and scheduler is not None:
                    postfix["lr"] = f"{scheduler.get_last_lr()[0]:.2e}"
                pbar.set_postfix(postfix)

            # Periodic checkpoint
            if (training and args.checkpoint_minutes > 0
                    and time.time() - last_ckpt_time >= args.checkpoint_minutes * 60):
                ckpt_path = os.path.join(args.output_dir, "checkpoint_latest")
                save_checkpoint(ckpt_path, predictor, optimizer, scheduler,
                                epoch, global_step, ma=ma)
                last_ckpt_time = time.time()

            if lr_exhausted:
                print(f"\n  LR reached zero at step {global_step}, stopping epoch early")
                break

        # Flush remaining accumulated gradients
        if training and not lr_exhausted and accum_count % args.grad_accum_steps != 0:
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
    return metrics, global_step, lr_exhausted


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

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print(f"\nLoading documents from {args.data}...")
    documents = torch.load(args.data, map_location="cpu", weights_only=False)

    print(f"Loading growth rates from {args.growth_rates}...")
    with open(args.growth_rates, "r") as f:
        growth_rates = json.load(f)
    print(f"  {len(growth_rates)} companies with growth rates")

    # Match documents to growth rates
    print("\nMatching documents to growth rates...")
    matched = match_documents_to_growth_rates(documents, growth_rates)
    del documents, growth_rates

    if not matched:
        print("ERROR: No documents matched to growth rates. Check company ID extraction.")
        return

    # Extract just the docs for splitting (split_documents needs dicts with source_file)
    matched_docs = [doc for doc, _ in matched]
    train_docs_raw, val_docs_raw = split_documents(matched_docs, args.val_ratio)
    del matched_docs

    # Rebuild matched pairs from split results
    rate_lookup = {id(doc): rate for doc, rate in matched}
    del matched
    train_data = [(doc, rate_lookup[id(doc)]) for doc in train_docs_raw]
    val_data = [(doc, rate_lookup[id(doc)]) for doc in val_docs_raw]
    del rate_lookup, train_docs_raw, val_docs_raw

    print(f"  {len(train_data)} train, {len(val_data)} val documents")

    # ------------------------------------------------------------------
    # Token info
    # ------------------------------------------------------------------
    tok_info = get_token_info(PRETRAINED_ID)

    # ------------------------------------------------------------------
    # Encoder (frozen)
    # ------------------------------------------------------------------
    print("\nLoading encoder...")
    encoder_model, config = load_encoder(args.encoder_checkpoint, device)

    # ------------------------------------------------------------------
    # GrowthPredictor (aggregator + head)
    # ------------------------------------------------------------------
    print("\nBuilding GrowthPredictor...")
    predictor = build_predictor(
        aggregator_checkpoint=args.aggregator_checkpoint,
        hidden_size=config.hidden_size,
        n_bins=args.n_bins,
        min_val=args.min_val,
        max_val=args.max_val,
        train_aggregator=args.train_aggregator,
    )
    predictor = predictor.to(device)

    if args.compile:
        print("Compiling predictor with torch.compile...")
        predictor = torch.compile(predictor, dynamic=True)

    # ------------------------------------------------------------------
    # Optimizer + schedule
    # ------------------------------------------------------------------
    param_groups = []

    # Head parameters (always trainable)
    unwrapped = predictor._orig_mod if hasattr(predictor, "_orig_mod") else predictor
    head_params = list(unwrapped.head.parameters())
    param_groups.append({
        "params": head_params,
        "lr": args.lr,
        "name": "head",
    })

    # Aggregator parameters (optionally trainable)
    if args.train_aggregator:
        agg_params = list(unwrapped.aggregator.parameters())
        param_groups.append({
            "params": agg_params,
            "lr": args.aggregator_lr,
            "name": "aggregator",
        })

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=args.weight_decay,
    )

    print("\nEstimating steps per epoch...")
    steps_per_epoch = estimate_steps_per_epoch(
        train_data, tok_info, args.seed, args.cls_budget, args.grad_accum_steps,
    )
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
        unwrapped = predictor._orig_mod if hasattr(predictor, "_orig_mod") else predictor
        unwrapped.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        print(f"  Resumed at epoch={start_epoch}, global_step={global_step}")

    # ------------------------------------------------------------------
    # Regularization: pre-compute reference aggregator outputs
    # ------------------------------------------------------------------
    reference_outputs = None
    if args.regularization and args.train_aggregator:
        print("\nPre-computing reference aggregator outputs for regularization...")
        # Need to chunk + CLS once to get references
        ref_chunks = chunk_documents(
            train_data, tok_info["newline_ids"],
            tok_info["cls_id"], tok_info["sep_id"], args.seed,
        )
        ref_cls = compute_all_cls(encoder_model, ref_chunks, device,
                                  tok_info["pad_id"], args.token_budget)
        del ref_chunks
        ref_doc_cls, ref_doc_rates = gather_doc_embeddings(ref_cls, train_data)
        del ref_cls

        unwrapped = predictor._orig_mod if hasattr(predictor, "_orig_mod") else predictor
        reference_outputs = compute_reference_outputs(
            unwrapped, ref_doc_cls, ref_doc_rates, args.cls_budget, device,
        )
        del ref_doc_cls, ref_doc_rates
        print(f"  Computed {len(reference_outputs)} reference embeddings")

    # ------------------------------------------------------------------
    # Print config
    # ------------------------------------------------------------------
    print(f"\nTraining: {args.epochs} epochs, ~{steps_per_epoch} steps/epoch, "
          f"~{total_steps} total steps")
    print(f"cls_budget: {args.cls_budget}, grad_accum: {args.grad_accum_steps}")
    print(f"Head LR: {args.lr}, warmup: {warmup_steps} steps")
    if args.train_aggregator:
        print(f"Aggregator LR: {args.aggregator_lr}")
    if args.regularization:
        print(f"Regularization: lambda={args.reg_lambda}")
    print(f"Bins: {args.n_bins}, range: [{args.min_val}, {args.max_val}]")
    print()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*70}")

        t0 = time.time()
        train_metrics, global_step, lr_exhausted = run_epoch(
            predictor, train_data, encoder_model, device, tok_info, args,
            optimizer=optimizer, scheduler=scheduler,
            epoch=epoch, global_step=global_step,
            training=True, reference_outputs=reference_outputs,
        )
        t_train = time.time() - t0

        print(f"\n  Train: loss={train_metrics['loss']:.4f} "
              f"(ce={train_metrics['ce']:.4f}, "
              f"reg={train_metrics['reg']:.4f}, "
              f"mae={train_metrics['mae']:.4f}) [{t_train:.0f}s]")

        # Validation
        val_loss = None
        if val_data:
            t0 = time.time()
            val_metrics, _, _ = run_epoch(
                predictor, val_data, encoder_model, device, tok_info, args,
                epoch=epoch, training=False,
            )
            t_val = time.time() - t0
            val_loss = val_metrics["loss"]
            print(f"  Val:   loss={val_metrics['loss']:.4f} "
                  f"(ce={val_metrics['ce']:.4f}, "
                  f"mae={val_metrics['mae']:.4f}) [{t_val:.0f}s]")

        # End-of-epoch checkpoint (weights only)
        ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch + 1}")
        save_checkpoint(ckpt_path, predictor, optimizer, scheduler,
                        epoch + 1, global_step,
                        val_loss=val_loss, args=args, weights_only=True)

        # Latest checkpoint (with optimizer state for resuming)
        latest_path = os.path.join(args.output_dir, "checkpoint_latest")
        save_checkpoint(latest_path, predictor, optimizer, scheduler,
                        epoch + 1, global_step,
                        val_loss=val_loss, args=args, weights_only=False)

        if lr_exhausted:
            break

    print("\nTraining complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Growth rate prediction training")

    # Data
    parser.add_argument("--data", required=True,
                        help="Path to documents.pt")
    parser.add_argument("--growth-rates", required=True,
                        help="Path to growth_rates.json ({company_id: rate})")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of documents for validation")

    # Checkpoints (input)
    parser.add_argument("--encoder-checkpoint", required=True,
                        help="Path to T5 checkpoint (full_model.pt)")
    parser.add_argument("--aggregator-checkpoint", type=str, default=None,
                        help="Path to aggregator checkpoint")

    # Model
    parser.add_argument("--n-bins", type=int, default=32,
                        help="Number of growth rate bins")
    parser.add_argument("--min-val", type=float, default=-1.0,
                        help="Minimum growth rate")
    parser.add_argument("--max-val", type=float, default=1.0,
                        help="Maximum growth rate")

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--cls-budget", type=int, default=256,
                        help="Max total CLS tokens per document batch")
    parser.add_argument("--token-budget", type=int, default=8192,
                        help="Token budget for CLS embedding batches")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for head")
    parser.add_argument("--aggregator-lr", type=float, default=1e-6,
                        help="Learning rate for aggregator (with --train-aggregator)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    # Aggregator training
    parser.add_argument("--train-aggregator", action="store_true",
                        help="Also train the aggregator (with separate LR)")
    parser.add_argument("--regularization", action="store_true",
                        help="Add MSE regularization on aggregator outputs")
    parser.add_argument("--reg-lambda", type=float, default=0.1,
                        help="Weight for regularization loss")

    # Output / resuming
    parser.add_argument("--output-dir", default="checkpoints/growth_predictor")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to checkpoint directory to resume from")
    parser.add_argument("--checkpoint-minutes", type=int, default=30,
                        help="Save periodic checkpoint every N minutes (0=disable)")

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
