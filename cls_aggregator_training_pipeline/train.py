"""
CLS Aggregator training: chunking → CLS encoding → aggregation → decoder denoising.

Per epoch:
1. Chunk each document twice with random strategies:
   - Encoder: consistent random target size per doc, segment-aware boundaries
   - Decoder: random target size per chunk, random boundaries
2. Sort encoder chunks by length, batch by token budget, compute CLS embeddings
3. Gather CLS embeddings per document → aggregator input
4. Sort decoder chunks by length, batch by token budget
5. Train: aggregator(doc_CLS) → single enriched CLS → decoder denoises chunk → loss

Encoder and decoder are frozen. Only the aggregator is trained.

Usage (multi-GPU):
    torchrun --nproc_per_node=4 -m cls_aggregator_training_pipeline.train \
        --data cls_aggregator_data/documents.pt \
        --checkpoint checkpoints/t5_expanded_memory/model_only.pt
"""
import argparse
import bisect
import math
import os
import random
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cls_aggregator_training_pipeline.aggregator import CLSAggregator
from financial_bert import FinancialBertTokenizer, FinancialModernBertConfig
from split_utils import split_documents
from t5_style_training_pipeline.decoder import T5StyleModel
from t5_style_training_pipeline.train import create_masked_inputs

PRETRAINED_ID = "answerdotai/ModernBERT-base"


def is_rank0():
    return not dist.is_initialized() or dist.get_rank() == 0


def log(msg):
    if is_rank0():
        print(msg)


CHUNK_MIN = 128
CHUNK_MAX = 1024
MIN_TAIL_CHUNK = 16  # absorb trailing chunks shorter than this


# ─── Model Loading ──────────────────────────────────────────────────────────

def load_t5_model(checkpoint_path, device):
    """Load T5 model from checkpoint, freeze all parameters."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = FinancialModernBertConfig.from_pretrained(PRETRAINED_ID)
    config.num_magnitude_bins = ckpt["args"].get("num_magnitude_bins", 128)

    model = T5StyleModel(config, cross_attn_type="expanded_memory")
    state_dict = {k.removeprefix("_orig_mod."): v
                  for k, v in ckpt["model_state_dict"].items()}
    del ckpt
    model.load_state_dict(state_dict)
    del state_dict

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log(f"  Loaded T5 model: {n_params:.1f}M params (frozen)")
    return model, config


def get_token_info(pretrained_id):
    """Extract special token IDs needed for training."""
    tokenizer = FinancialBertTokenizer(pretrained_id)
    base = tokenizer.base_tokenizer
    newline_ids = set(base.encode("\n", add_special_tokens=False))
    info = {
        "cls_id": base.cls_token_id,
        "sep_id": base.sep_token_id,
        "pad_id": base.pad_token_id,
        "mask_id": tokenizer.mask_token_id,
        "newline_ids": newline_ids,
    }
    del tokenizer
    return info


# ─── Chunking ───────────────────────────────────────────────────────────────

def get_boundaries(input_ids, newline_ids):
    """Find chunk boundary candidates (positions after newline tokens).

    Works in content space (CLS/SEP stripped: position 0 = first content token).
    Returns sorted list of boundary positions.
    """
    content = input_ids[1:-1]  # strip CLS at 0 and SEP at -1
    mask = torch.zeros(len(content), dtype=torch.bool)
    for nl_id in newline_ids:
        mask |= (content == nl_id)
    # Boundary = position AFTER the newline (i.e. start of next chunk)
    return (mask.nonzero(as_tuple=False).squeeze(-1) + 1).tolist()


def snap_to_boundary(target, boundaries, min_pos):
    """Snap a target position to the nearest boundary candidate."""
    if not boundaries:
        return target
    idx = bisect.bisect_left(boundaries, target)
    candidates = []
    if idx > 0 and boundaries[idx - 1] > min_pos:
        candidates.append(boundaries[idx - 1])
    if idx < len(boundaries):
        candidates.append(boundaries[idx])
    if not candidates:
        return target
    return min(candidates, key=lambda b: abs(b - target))


def chunk_spans(content_len, boundaries, size_fn):
    """Generate (start, end) chunk spans for a document's content."""
    spans = []
    pos = 0
    while pos < content_len:
        target = size_fn()
        raw_end = min(pos + target, content_len)

        if raw_end < content_len:
            end = snap_to_boundary(raw_end, boundaries, pos)
            if end <= pos:
                end = min(pos + target, content_len)
            # Absorb tiny tail
            if content_len - end < MIN_TAIL_CHUNK:
                end = content_len
        else:
            end = content_len

        spans.append((pos, end))
        pos = end
    return spans


def extract_chunk(doc, start, end, cls_id, sep_id):
    """Extract a chunk from document content (content space), wrap with CLS/SEP."""
    offset = 1 + start  # +1 to skip document CLS
    length = end - start

    c_ids = doc["input_ids"][offset:offset + length]
    c_mask = doc["is_number_mask"][offset:offset + length]
    c_vals = doc["number_values"][offset:offset + length]

    chunk_ids = torch.cat([
        torch.tensor([cls_id], dtype=c_ids.dtype), c_ids,
        torch.tensor([sep_id], dtype=c_ids.dtype),
    ])
    chunk_mask = torch.cat([
        torch.tensor([False]), c_mask.bool(),
        torch.tensor([False]),
    ])
    chunk_vals = torch.cat([
        torch.tensor([0.0]), c_vals.float(),
        torch.tensor([0.0]),
    ])

    return {
        "input_ids": chunk_ids,
        "is_number_mask": chunk_mask,
        "number_values": chunk_vals,
        "seq_length": len(chunk_ids),
        "doc_start": start,
        "doc_end": end,
    }


# ─── Batching ───────────────────────────────────────────────────────────────

def form_batches(items, token_budget):
    """Group length-sorted items into batches (max_len × batch_size ≤ budget)."""
    batches = []
    current = []
    current_max = 0

    for item in items:
        l = item["seq_length"]
        new_max = max(current_max, l)
        if current and new_max * (len(current) + 1) > token_budget:
            batches.append(current)
            current = [item]
            current_max = l
        else:
            current.append(item)
            current_max = new_max

    if current:
        batches.append(current)
    return batches


def pad_and_collate(chunks, pad_id):
    """Pad and stack chunk dicts into batch tensors."""
    max_len = max(c["seq_length"] for c in chunks)
    B = len(chunks)

    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    is_number_mask = torch.zeros(B, max_len, dtype=torch.long)
    number_values = torch.zeros(B, max_len, dtype=torch.float32)
    attention_mask = torch.zeros(B, max_len, dtype=torch.long)

    for i, c in enumerate(chunks):
        l = c["seq_length"]
        input_ids[i, :l] = c["input_ids"]
        is_number_mask[i, :l] = c["is_number_mask"].long()
        number_values[i, :l] = c["number_values"].float()
        attention_mask[i, :l] = 1

    return input_ids, is_number_mask, number_values, attention_mask


# ─── CLS Computation ───────────────────────────────────────────────────────

@torch.no_grad()
def compute_all_cls(model, enc_chunks, device, token_budget, pad_id):
    """Compute CLS embeddings for all encoder chunks (replicated on every rank).

    No sharding — each rank processes all batches independently.  This avoids
    the massive all_gather_object that OOMs when serializing hundreds of
    thousands of CLS vectors via pickle.

    Returns: dict of doc_idx → Tensor(N, D) on CPU.
    """
    batches = form_batches(enc_chunks, token_budget)

    all_cls = {}  # (doc_idx, chunk_idx) → (D,)

    pbar = tqdm(batches, desc="  CLS embeddings", unit="batch",
                disable=not is_rank0())
    for batch in pbar:
        ids, num_mask, num_vals, attn_mask = pad_and_collate(batch, pad_id)
        ids = ids.to(device)
        num_mask = num_mask.to(device)
        num_vals = num_vals.to(device)
        attn_mask = attn_mask.to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            embeds = model._build_embeds(
                ids, num_vals, num_mask,
                model._get_encoder_tok_embeddings(),
                model.encoder.number_embedder,
            )
            out = model.encoder.modernbert(
                inputs_embeds=embeds, attention_mask=attn_mask,
            )
        cls_vecs = out.last_hidden_state[:, 0, :].float().cpu()

        for i, chunk in enumerate(batch):
            all_cls[(chunk["doc_idx"], chunk["chunk_idx"])] = cls_vecs[i]

    # Assemble per-document, ordered by chunk_idx
    doc_n = defaultdict(int)
    for d, c in all_cls:
        doc_n[d] = max(doc_n[d], c + 1)

    doc_cls = {}
    for d, n in doc_n.items():
        doc_cls[d] = torch.stack([all_cls[(d, i)] for i in range(n)])

    return doc_cls


# ─── Loss ───────────────────────────────────────────────────────────────────

def compute_loss(text_logits, mag_logits, labels_text, labels_magnitude, config):
    """Combined text CE + magnitude bin loss. Returns (total, text, mag)."""
    loss_text = F.cross_entropy(
        text_logits.view(-1, config.vocab_size),
        labels_text.view(-1),
        ignore_index=-100,
    )

    valid = labels_magnitude.view(-1) != -100
    if valid.any():
        targets = labels_magnitude.view(-1)[valid]
        preds = mag_logits.view(-1, config.num_magnitude_bins)[valid]

        mn, mx = config.magnitude_min, config.magnitude_max
        n_bins = config.num_magnitude_bins
        norm = (targets.clamp(mn, mx) - mn) / (mx - mn) * (n_bins - 1)
        lo = norm.floor().long()
        hi = norm.ceil().long()
        w_hi = norm - lo.float()
        w_lo = 1.0 - w_hi

        log_p = F.log_softmax(preds, dim=-1)
        loss_mag = -(
            w_lo * log_p.gather(1, lo.unsqueeze(1)).squeeze(1)
            + w_hi * log_p.gather(1, hi.unsqueeze(1)).squeeze(1)
        ).mean()
    else:
        loss_mag = torch.tensor(0.0, device=text_logits.device)

    return loss_text + loss_mag, loss_text, loss_mag


# ─── LR Schedule ────────────────────────────────────────────────────────────

def get_cosine_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─── Epoch ──────────────────────────────────────────────────────────────────

def run_epoch(aggregator, model, documents, optimizer, scheduler,
              device, epoch, args, config, tok_info, training=True):
    """Run one training or validation epoch."""
    prefix = "Train" if training else "Val"
    if training:
        aggregator.train()
    else:
        aggregator.eval()

    # Fixed seed per epoch (deterministic for val)
    rng_state = random.getstate()
    random.seed(args.seed + epoch + (0 if training else 10000))

    cls_id = tok_info["cls_id"]
    sep_id = tok_info["sep_id"]
    pad_id = tok_info["pad_id"]
    mask_id = tok_info["mask_id"]
    nl_ids = tok_info["newline_ids"]
    mag_sentinel = config.magnitude_max + 1.0

    # ─── Step 1: Chunk all documents ────────────────────────────────────
    all_enc_chunks = []
    all_dec_chunks = []
    doc_enc_spans = {}

    for doc_idx, doc in enumerate(documents):
        content_len = len(doc["input_ids"]) - 2  # strip CLS/SEP
        if content_len < 1:
            continue

        boundaries = get_boundaries(doc["input_ids"], nl_ids)

        # Encoder: consistent target per doc
        enc_target = random.randint(CHUNK_MIN, CHUNK_MAX)
        enc_spans = chunk_spans(content_len, boundaries, lambda: enc_target)
        doc_enc_spans[doc_idx] = enc_spans

        for ci, (s, e) in enumerate(enc_spans):
            chunk = extract_chunk(doc, s, e, cls_id, sep_id)
            chunk["doc_idx"] = doc_idx
            chunk["chunk_idx"] = ci
            all_enc_chunks.append(chunk)

        # Decoder: random target per chunk
        dec_spans = chunk_spans(
            content_len, boundaries, lambda: random.randint(CHUNK_MIN, CHUNK_MAX))
        for ci, (s, e) in enumerate(dec_spans):
            chunk = extract_chunk(doc, s, e, cls_id, sep_id)
            chunk["doc_idx"] = doc_idx
            all_dec_chunks.append(chunk)

    n_enc = len(all_enc_chunks)
    n_dec = len(all_dec_chunks)
    avg_enc = sum(c["seq_length"] for c in all_enc_chunks) / max(n_enc, 1)
    avg_dec = sum(c["seq_length"] for c in all_dec_chunks) / max(n_dec, 1)
    log(f"  Chunks: {n_enc} encoder (avg {avg_enc:.0f}), "
        f"{n_dec} decoder (avg {avg_dec:.0f})")

    # ─── Step 2: Compute CLS embeddings ─────────────────────────────────
    all_enc_chunks.sort(key=lambda c: c["seq_length"])
    doc_cls = compute_all_cls(
        model, all_enc_chunks, device, args.encoder_token_budget, pad_id)
    del all_enc_chunks

    # ─── Step 3: Sort decoder chunks, form batches, shard across ranks ──
    all_dec_chunks.sort(key=lambda c: c["seq_length"])
    all_dec_batches = form_batches(all_dec_chunks, args.decoder_token_budget)
    del all_dec_chunks

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    # Equalize batch counts across ranks (drop at most 1 batch)
    n_per_rank = len(all_dec_batches) // world_size
    dec_batches = all_dec_batches[rank::world_size][:n_per_rank]
    log(f"  {len(all_dec_batches)} decoder batches total, "
        f"{n_per_rank} per rank")

    # ─── Step 4: Train/eval on decoder batches ──────────────────────────
    total_loss = 0.0
    total_loss_text = 0.0
    total_loss_mag = 0.0
    total_correct = 0
    total_masked = 0
    n_batches = 0
    accum_count = 0

    prev_grad = torch.is_grad_enabled()
    if not training:
        torch.set_grad_enabled(False)

    if training:
        optimizer.zero_grad()

    try:
        pbar = tqdm(dec_batches, desc=f"  {prefix}", unit="batch",
                    disable=not is_rank0())
        for batch_idx, batch_chunks in enumerate(pbar):
            # Aggregator: one forward per unique document in this batch
            doc_indices = set(c["doc_idx"] for c in batch_chunks)
            enriched_cache = {}

            with torch.autocast("cuda", dtype=torch.bfloat16):
                for d in doc_indices:
                    raw = doc_cls[d].to(device)  # (N, D)
                    enriched_cache[d] = aggregator(raw.unsqueeze(0)).squeeze(0)  # (D,)

                # Stack single CLS for each decoder chunk in batch
                batch_cls = torch.stack(
                    [enriched_cache[c["doc_idx"]] for c in batch_chunks])  # (B, D)

                # Pad decoder batch
                ids, num_mask, num_vals, attn_mask = pad_and_collate(
                    batch_chunks, pad_id)
                ids = ids.to(device)
                num_mask = num_mask.to(device)
                num_vals = num_vals.to(device)
                attn_mask = attn_mask.to(device)

                # Apply masking
                (m_ids, m_nums, m_num_mask,
                 labels_t, labels_m) = create_masked_inputs(
                    ids, num_mask, num_vals,
                    mask_token_id=mask_id, pad_token_id=pad_id,
                    magnitude_sentinel=mag_sentinel,
                    vocab_size=config.vocab_size,
                    magnitude_min=config.magnitude_min,
                    magnitude_max=config.magnitude_max,
                    mask_prob_min=args.mask_prob_min,
                    mask_prob_max=args.mask_prob_max,
                )

                # Decoder forward
                dec_embeds = model._build_embeds(
                    m_ids, m_nums, m_num_mask,
                    model.decoder._get_tok_embeddings(),
                    model.decoder.number_embedder,
                )
                text_logits, mag_logits = model.decoder(
                    dec_embeds, batch_cls, attn_mask)

            # Loss in fp32
            text_logits = text_logits.float()
            mag_logits = mag_logits.float()
            loss, lt, lm = compute_loss(
                text_logits, mag_logits, labels_t, labels_m, config)

            if training:
                accum_count += 1
                is_sync_step = accum_count % args.grad_accum_steps == 0
                sync_ctx = nullcontext() if is_sync_step else aggregator.no_sync()
                with sync_ctx:
                    (loss / args.grad_accum_steps).backward()

                if is_sync_step:
                    torch.nn.utils.clip_grad_norm_(
                        aggregator.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            # Metrics
            total_loss += loss.item()
            total_loss_text += lt.item()
            total_loss_mag += lm.item()

            text_masked = labels_t.view(-1) != -100
            if text_masked.any():
                preds = text_logits.detach().view(
                    -1, config.vocab_size)[text_masked].argmax(-1)
                total_correct += (preds == labels_t.view(-1)[text_masked]).sum().item()
                total_masked += text_masked.sum().item()

            n_batches += 1

            if n_batches % 20 == 0:
                avg_l = total_loss / n_batches
                acc = total_correct / max(total_masked, 1)
                lr = optimizer.param_groups[0]["lr"] if training else 0
                pbar.set_postfix(
                    loss=f"{avg_l:.3f}", acc=f"{acc:.1%}", lr=f"{lr:.2e}")

        # Flush remaining accumulated gradients (all ranks have same count,
        # so either all flush or none)
        if training and accum_count % args.grad_accum_steps != 0:
            # Manual allreduce since we used no_sync for these steps
            for p in aggregator.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            torch.nn.utils.clip_grad_norm_(
                aggregator.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    finally:
        torch.set_grad_enabled(prev_grad)
        random.setstate(rng_state)

    # Reduce metrics across ranks
    if dist.is_initialized() and dist.get_world_size() > 1:
        stats = torch.tensor(
            [total_loss, total_loss_text, total_loss_mag,
             total_correct, total_masked, n_batches],
            dtype=torch.float64, device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss, total_loss_text, total_loss_mag, \
            total_correct, total_masked, n_batches = stats.tolist()
        n_batches = int(n_batches)

    avg_loss = total_loss / max(n_batches, 1)
    avg_lt = total_loss_text / max(n_batches, 1)
    avg_lm = total_loss_mag / max(n_batches, 1)
    acc = total_correct / max(total_masked, 1)

    return {
        "loss": avg_loss,
        "loss_text": avg_lt,
        "loss_mag": avg_lm,
        "text_acc": acc,
        "n_batches": n_batches,
    }


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train CLS Aggregator")
    parser.add_argument("--data", default="cls_aggregator_data/documents.pt")
    parser.add_argument("--checkpoint",
                        default="checkpoints/t5_expanded_memory/model_only.pt")
    parser.add_argument("--output_dir", default="checkpoints/cls_aggregator")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--encoder_token_budget", type=int, default=32768)
    parser.add_argument("--decoder_token_budget", type=int, default=16384)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--mask_prob_min", type=float, default=0.15)
    parser.add_argument("--mask_prob_max", type=float, default=0.50)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    # Aggregator hyperparams
    parser.add_argument("--agg_layers", type=int, default=6)
    parser.add_argument("--agg_heads", type=int, default=16)
    parser.add_argument("--agg_hidden", type=int, default=768)
    parser.add_argument("--agg_dropout", type=float, default=0.1)
    args = parser.parse_args()

    # Distributed setup
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    log(f"World size: {dist.get_world_size()}, device: {device}")

    # Load T5 model (frozen, replicated on each GPU)
    log(f"\nLoading T5 model from {args.checkpoint}...")
    model, config = load_t5_model(args.checkpoint, device)

    # Token info
    tok_info = get_token_info(PRETRAINED_ID)

    # Load documents
    log(f"\nLoading documents from {args.data}...")
    documents = torch.load(args.data, map_location="cpu", weights_only=False)
    train_docs, val_docs = split_documents(documents, args.val_ratio)
    del documents
    log(f"  {len(train_docs)} train, {len(val_docs)} val documents")

    # Build aggregator (wrapped in DDP)
    aggregator = CLSAggregator(
        hidden_size=args.agg_hidden,
        num_heads=args.agg_heads,
        num_layers=args.agg_layers,
        dropout=args.agg_dropout,
    ).to(device)
    aggregator = DDP(aggregator, device_ids=[local_rank])
    n_agg = sum(p.numel() for p in aggregator.parameters()) / 1e6
    log(f"\nAggregator: {n_agg:.1f}M params (DDP)")

    # Optimizer & schedule
    optimizer = torch.optim.AdamW(
        aggregator.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    world_size = dist.get_world_size()
    total_content = sum(len(d["input_ids"]) - 2 for d in train_docs)
    est_batches = total_content / args.decoder_token_budget
    est_batches_per_rank = est_batches / world_size
    total_steps = int(est_batches_per_rank * args.epochs / args.grad_accum_steps)

    scheduler = get_cosine_schedule(optimizer, args.warmup_steps, total_steps)
    log(f"Estimated ~{est_batches:.0f} batches/epoch "
        f"({est_batches_per_rank:.0f}/rank), {total_steps} total steps")

    # Resume
    start_epoch = 0
    os.makedirs(args.output_dir, exist_ok=True)

    if args.resume:
        latest = os.path.join(args.output_dir, "latest.pt")
        if os.path.exists(latest):
            ckpt = torch.load(latest, map_location="cpu", weights_only=False)
            aggregator.module.load_state_dict(ckpt["aggregator"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            log(f"Resumed from epoch {start_epoch}")
            del ckpt

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        log(f"\n{'='*70}")
        log(f"Epoch {epoch + 1}/{args.epochs}")
        log(f"{'='*70}")

        # Train
        t0 = time.time()
        train_m = run_epoch(
            aggregator, model, train_docs, optimizer, scheduler,
            device, epoch, args, config, tok_info, training=True)
        t_train = time.time() - t0

        log(f"\n  Train: loss={train_m['loss']:.4f} "
            f"(text={train_m['loss_text']:.4f}, mag={train_m['loss_mag']:.4f}) "
            f"acc={train_m['text_acc']:.1%} [{t_train:.0f}s]")

        # Validate
        t0 = time.time()
        val_m = run_epoch(
            aggregator, model, val_docs, optimizer, scheduler,
            device, epoch, args, config, tok_info, training=False)
        t_val = time.time() - t0

        log(f"  Val:   loss={val_m['loss']:.4f} "
            f"(text={val_m['loss_text']:.4f}, mag={val_m['loss_mag']:.4f}) "
            f"acc={val_m['text_acc']:.1%} [{t_val:.0f}s]")

        # Save checkpoint (rank 0 only, unwrapped state dict)
        if is_rank0():
            ckpt = {
                "aggregator": aggregator.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "train_metrics": train_m,
                "val_metrics": val_m,
                "args": vars(args),
            }
            torch.save(ckpt, os.path.join(args.output_dir, "latest.pt"))
            torch.save(ckpt, os.path.join(
                args.output_dir, f"epoch_{epoch + 1}.pt"))
            log(f"  Saved checkpoint: epoch_{epoch + 1}.pt")

        # Barrier so all ranks wait for checkpoint save before next epoch
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
