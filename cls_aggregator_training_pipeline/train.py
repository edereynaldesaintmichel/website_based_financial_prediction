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
6. Contrastive loss: same document with 2 dropout masks → positive pair,
   different documents → negatives (InfoNCE)

Encoder and decoder are frozen. Only the aggregator is trained.

Usage (single GPU):
    python -m cls_aggregator_training_pipeline.train \
        --data mlm_data/documents.pt \
        --checkpoint checkpoints/t5_cls/checkpoint_epoch5/full_model.pt

Usage (multi-GPU via DDP):
    torchrun --nproc_per_node=4 -m cls_aggregator_training_pipeline.train \
        --data mlm_data/documents.pt \
        --checkpoint checkpoints/t5_cls/checkpoint_epoch5/full_model.pt
"""
import argparse
import math
import os
import random
import sys
import time
from collections import defaultdict, deque
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cls_aggregator_training_pipeline.aggregator import CLSAggregator
from chunk_utils import (
    get_token_info,
    get_boundaries,
    snap_to_boundary,
    chunk_spans,
    extract_chunk,
    pad_and_collate,
)
from financial_bert import FinancialBertTokenizer, FinancialModernBertConfig
from split_utils import split_documents
from t5_style_training_pipeline.decoder import T5StyleModel
from t5_style_training_pipeline.train import create_masked_inputs

PRETRAINED_ID = "answerdotai/ModernBERT-base"


CHUNK_MIN = 128
CHUNK_MAX = 1024


# ─── Distributed Helpers ──────────────────────────────────────────────────

def setup_distributed():
    """Detect torchrun env and init process group. Returns (rank, world_size).

    Returns (0, 1) when running without torchrun (single-GPU).
    """
    if "RANK" not in os.environ:
        return 0, 1
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def rprint(rank, *args, **kwargs):
    """Print only on rank 0."""
    if rank == 0:
        print(*args, **kwargs)

# ─── Model Loading ──────────────────────────────────────────────────────────

def load_t5_model(checkpoint_path, device, rank=0):
    """Load T5 model from checkpoint. Encoder is frozen, decoder is trainable."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = FinancialModernBertConfig.from_pretrained(PRETRAINED_ID)
    config.num_magnitude_bins = ckpt["args"].get("num_magnitude_bins", 128)

    model = T5StyleModel(config)
    state_dict = {k.removeprefix("_orig_mod."): v
                  for k, v in ckpt["model_state_dict"].items()}
    del ckpt
    model.load_state_dict(state_dict)
    del state_dict

    # Freeze encoder, leave decoder trainable
    model.encoder.eval()
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    model.to(device)

    n_enc = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    n_dec = sum(p.numel() for p in model.decoder.parameters()) / 1e6
    rprint(rank, f"  Loaded T5 model: encoder {n_enc:.1f}M (frozen), "
                 f"decoder {n_dec:.1f}M (trainable)")
    return model, config


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


def form_batches_shuffled(items, token_budget, bucket_width=16):
    """Bucket items by length, form batches per bucket, shuffle batch order.

    Like form_batches but without systematic short→long ordering.
    Padding overhead is bounded by bucket_width per sequence.
    """
    buckets = defaultdict(list)
    for item in items:
        key = (item["seq_length"] + bucket_width - 1) // bucket_width * bucket_width
        buckets[key].append(item)

    all_batches = []
    for bucket_len, bucket_items in buckets.items():
        random.shuffle(bucket_items)
        batch_size = max(1, token_budget // bucket_len)
        for i in range(0, len(bucket_items), batch_size):
            all_batches.append(bucket_items[i:i + batch_size])

    random.shuffle(all_batches)
    return all_batches


AGG_COST_WEIGHT = 6    # relative memory cost per aggregator token
DEC_COST_WEIGHT = 30   # relative memory cost per decoder token


def form_batches_2d(items, budget, bucket_width=16):
    """Batch decoder chunks using 2D bucketing (dec_len × n_cls).

    Groups chunks from the same document together and accounts for
    aggregator memory cost (per unique document) separately from
    decoder memory cost (per chunk).

    Budget constraint per batch:
        AGG_COST_WEIGHT * n_unique_docs * max_n_cls
        + DEC_COST_WEIGHT * n_chunks * max_dec_len ≤ budget

    Each item must have 'seq_length', 'n_cls', and 'doc_idx' fields.
    """
    # Bucket by decoder length (rows)
    buckets = defaultdict(list)
    for item in items:
        key = (item["seq_length"] + bucket_width - 1) // bucket_width * bucket_width
        buckets[key].append(item)

    all_batches = []
    for bucket_dec_len, bucket_items in buckets.items():
        # Sort by (n_cls, doc_idx) — groups same-doc chunks, orders by agg cost
        bucket_items.sort(key=lambda c: (c["n_cls"], c["doc_idx"]))

        # Greedily form batches
        current_batch = []
        current_docs = set()
        current_max_n_cls = 0

        for chunk in bucket_items:
            doc_idx = chunk["doc_idx"]
            n_cls = chunk["n_cls"]

            # Cost if we add this chunk
            new_unique = len(current_docs | {doc_idx})
            new_max_n_cls = max(current_max_n_cls, n_cls)
            new_n_chunks = len(current_batch) + 1

            cost = (AGG_COST_WEIGHT * new_unique * new_max_n_cls
                    + DEC_COST_WEIGHT * new_n_chunks * bucket_dec_len)

            if current_batch and cost > budget:
                all_batches.append(current_batch)
                current_batch = [chunk]
                current_docs = {doc_idx}
                current_max_n_cls = n_cls
            else:
                current_batch.append(chunk)
                current_docs.add(doc_idx)
                current_max_n_cls = new_max_n_cls

        if current_batch:
            all_batches.append(current_batch)

    random.shuffle(all_batches)
    return all_batches


# ─── CLS Computation ───────────────────────────────────────────────────────

@torch.no_grad()
def compute_all_cls(model, enc_chunks, device, token_budget, pad_id,
                    rank=0, world_size=1, cache_path=None):
    """Compute CLS embeddings for all encoder chunks.

    In multi-GPU mode, each rank processes its shard of batches, saves to a
    temporary file, and rank 0 merges all shards into the final cache.
    This avoids the memory spike of all-gathering large result dicts.

    Returns: dict of doc_idx → Tensor(N, D) on CPU.
    """
    batches = form_batches(enc_chunks, token_budget)
    # Each rank takes its slice of batches
    my_batches = batches[rank::world_size]

    hidden_size = None
    results = {}  # doc_idx → {chunk_idx → cls_vec}

    pbar = tqdm(my_batches, desc="  CLS embeddings", unit="batch",
                disable=(rank != 0))
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

        if hidden_size is None:
            hidden_size = cls_vecs.shape[1]

        for i, chunk in enumerate(batch):
            doc_idx = chunk["doc_idx"]
            chunk_idx = chunk["chunk_idx"]
            if doc_idx not in results:
                results[doc_idx] = {}
            results[doc_idx][chunk_idx] = cls_vecs[i]

    # Free GPU memory from encoder forward passes
    torch.cuda.empty_cache()

    # Multi-GPU: merge via disk instead of all-gather to avoid memory spike
    if world_size > 1 and cache_path is not None:
        shard_path = cache_path + f".shard_{rank}"
        torch.save(results, shard_path)
        del results
        dist.barrier()

        if rank == 0:
            merged = {}
            for r in range(world_size):
                shard = torch.load(
                    cache_path + f".shard_{r}",
                    map_location="cpu", weights_only=False)
                for doc_idx, chunks_dict in shard.items():
                    if doc_idx not in merged:
                        merged[doc_idx] = {}
                    merged[doc_idx].update(chunks_dict)
                del shard
            results = merged
        else:
            # Non-zero ranks will load the final cache after rank 0 saves it
            return None
    elif world_size > 1:
        # Fallback if no cache_path: use all-gather
        all_results = [None] * world_size
        dist.all_gather_object(all_results, results)
        merged = {}
        for r in all_results:
            for doc_idx, chunks_dict in r.items():
                if doc_idx not in merged:
                    merged[doc_idx] = {}
                merged[doc_idx].update(chunks_dict)
        results = merged

    if hidden_size is None:
        hidden_size = 768  # fallback

    # Assemble per-document tensors ordered by chunk_idx
    doc_cls = {}
    for doc_idx, chunks_dict in results.items():
        n = max(chunks_dict.keys()) + 1
        tensor = torch.zeros(n, hidden_size)
        for ci, vec in chunks_dict.items():
            tensor[ci] = vec
        doc_cls[doc_idx] = tensor

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
        loss_mag = 0.0 * mag_logits.sum()  # keep in graph

    return loss_text + loss_mag, loss_text, loss_mag


def compute_contrastive_loss(z1, z2, temperature=0.05):
    """InfoNCE contrastive loss between two views of the same documents.

    Args:
        z1, z2: (N, D) L2-normalized embeddings from two dropout-augmented
                forward passes of the aggregator on the same documents.
        temperature: softmax temperature.
    Returns:
        Scalar InfoNCE loss.
    """
    # (N, N) similarity matrix
    logits = z1 @ z2.T / temperature  # (N, N)
    labels = torch.arange(z1.shape[0], device=z1.device)
    # Symmetric: average both directions
    loss = (F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.T, labels)) * 0.5
    return loss


def evaluate_contrastive_positives(dec_batches):
    """Report unique-documents-per-batch stats (= contrastive positive count).

    Each unique document in a batch produces one SimCSE positive pair
    (same doc, two dropout masks). Batches with ≤1 doc yield no contrastive signal.
    """
    docs_per_batch = []
    for batch in dec_batches:
        n_unique = len(set(c["doc_idx"] for c in batch))
        docs_per_batch.append(n_unique)

    n_batches = len(docs_per_batch)
    n_contrastive = sum(1 for d in docs_per_batch if d > 1)
    avg_docs = sum(docs_per_batch) / max(n_batches, 1)
    min_docs = min(docs_per_batch) if docs_per_batch else 0
    max_docs = max(docs_per_batch) if docs_per_batch else 0

    return avg_docs, min_docs, max_docs, n_contrastive, n_batches


# ─── LR Schedule ────────────────────────────────────────────────────────────

def get_cosine_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─── Epoch ──────────────────────────────────────────────────────────────────

def run_epoch(aggregator, decoder, model, documents, optimizer, scheduler,
              device, epoch, args, config, tok_info, training=True,
              rank=0, world_size=1):
    """Run one training or validation epoch."""
    prefix = "Train" if training else "Val"
    contrastive_lambda = getattr(args, "contrastive_lambda", 0.0)
    contrastive_temp = getattr(args, "contrastive_temp", 0.05)
    if training:
        aggregator.train()
        decoder.train()
    else:
        aggregator.eval()
        decoder.eval()

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
        n_cls = len(enc_spans)
        for ci, (s, e) in enumerate(dec_spans):
            chunk = extract_chunk(doc, s, e, cls_id, sep_id)
            chunk["doc_idx"] = doc_idx
            chunk["n_cls"] = n_cls
            all_dec_chunks.append(chunk)

    n_enc = len(all_enc_chunks)
    n_dec = len(all_dec_chunks)
    avg_enc = sum(c["seq_length"] for c in all_enc_chunks) / max(n_enc, 1)
    avg_dec = sum(c["seq_length"] for c in all_dec_chunks) / max(n_dec, 1)
    rprint(rank, f"  Chunks: {n_enc} encoder (avg {avg_enc:.0f}), "
                 f"{n_dec} decoder (avg {avg_dec:.0f})")

    # ─── Step 2: Compute CLS embeddings (cached to disk) ────────────────
    cache_tag = "train" if training else "val"
    cls_cache_path = os.path.join(
        args.output_dir, f"cls_cache_{cache_tag}_epoch{epoch}.pt")

    if os.path.exists(cls_cache_path):
        rprint(rank, f"  Loading cached CLS embeddings from {cls_cache_path}")
        doc_cls = torch.load(cls_cache_path, map_location="cpu", weights_only=False)
        del all_enc_chunks
    else:
        all_enc_chunks.sort(key=lambda c: c["seq_length"])
        doc_cls = compute_all_cls(
            model, all_enc_chunks, device, args.encoder_token_budget, pad_id,
            rank=rank, world_size=world_size, cache_path=cls_cache_path)
        del all_enc_chunks
        if rank == 0:
            torch.save(doc_cls, cls_cache_path)
            rprint(rank, f"  Cached CLS embeddings to {cls_cache_path}")
            # Clean up shard files
            for r in range(world_size):
                shard = cls_cache_path + f".shard_{r}"
                if os.path.exists(shard):
                    os.remove(shard)
        if world_size > 1:
            dist.barrier()  # Wait for rank 0 to save the merged cache
            if doc_cls is None:
                doc_cls = torch.load(cls_cache_path, map_location="cpu",
                                     weights_only=False)
    torch.cuda.empty_cache()

    # ─── Step 3: 2D bucketing (dec_len × n_cls), form batches ───────────
    avg_cls_per_doc = n_enc / max(len(doc_enc_spans), 1)
    batch_budget = (DEC_COST_WEIGHT * args.decoder_token_budget
                    + AGG_COST_WEIGHT * avg_cls_per_doc)
    rprint(rank, f"  Batch budget: {batch_budget:.0f} "
                 f"(avg {avg_cls_per_doc:.1f} CLS/doc)")
    dec_batches = form_batches_2d(all_dec_chunks, batch_budget)
    del all_dec_chunks

    # Padding waste estimation
    total_dec_useful = 0
    total_dec_padded = 0
    total_agg_useful = 0
    total_agg_padded = 0
    for batch in dec_batches:
        # Decoder: each chunk padded to max length in batch
        max_dec = max(c["seq_length"] for c in batch)
        total_dec_useful += sum(c["seq_length"] for c in batch)
        total_dec_padded += max_dec * len(batch)
        # Aggregator: unique docs padded to max n_cls in batch
        docs = {}
        for c in batch:
            docs[c["doc_idx"]] = c["n_cls"]
        max_cls = max(docs.values())
        total_agg_useful += sum(docs.values())
        total_agg_padded += max_cls * len(docs)
    dec_waste = 1.0 - total_dec_useful / max(total_dec_padded, 1)
    agg_waste = 1.0 - total_agg_useful / max(total_agg_padded, 1)
    weighted_waste = (
        (DEC_COST_WEIGHT * (total_dec_padded - total_dec_useful)
         + AGG_COST_WEIGHT * (total_agg_padded - total_agg_useful))
        / max(DEC_COST_WEIGHT * total_dec_padded
              + AGG_COST_WEIGHT * total_agg_padded, 1)
    )
    rprint(rank, f"  Padding waste: decoder {dec_waste:.1%}, "
                 f"aggregator {agg_waste:.1%}, "
                 f"weighted {weighted_waste:.1%}")
    rprint(rank, f"  {len(dec_batches)} decoder batches")

    # ─── Step 3b: Contrastive positive stats ──────────────────────────────
    avg_docs, min_docs, max_docs, n_cl, n_total = evaluate_contrastive_positives(
        dec_batches)
    rprint(rank, f"  Contrastive positives: {avg_docs:.1f} avg docs/batch "
                 f"(min {min_docs}, max {max_docs}), "
                 f"{n_cl}/{n_total} batches with ≥2 docs")

    # ─── Step 3c: Shard batches across ranks ──────────────────────────────
    dec_batches = dec_batches[rank::world_size]

    # ─── Step 3d: Pre-collate decoder batches into pinned memory ─────────
    dec_collated = []
    for batch in dec_batches:
        ids, num_mask, num_vals, attn_mask = pad_and_collate(batch, pad_id)
        dec_collated.append((
            ids.pin_memory(),
            num_mask.pin_memory(),
            num_vals.pin_memory(),
            attn_mask.pin_memory(),
        ))

    # ─── Step 4: Train/eval on decoder batches ──────────────────────────
    # Unwrap DDP for attribute access (forward still goes through DDP wrapper)
    decoder_inner = decoder.module if isinstance(decoder, DDP) else decoder

    total_loss = 0.0
    total_loss_text = 0.0
    total_loss_mag = 0.0
    total_loss_contrastive = 0.0
    total_correct = 0
    total_masked = 0
    n_batches = 0
    accum_count = 0
    recent_loss_text = deque(maxlen=100)
    recent_loss_mag = deque(maxlen=100)
    recent_loss_contrastive = deque(maxlen=100)

    prev_grad = torch.is_grad_enabled()
    if not training:
        torch.set_grad_enabled(False)

    if training:
        optimizer.zero_grad()

    try:
        pbar = tqdm(zip(dec_batches, dec_collated), total=len(dec_batches),
                    desc=f"  {prefix}", unit="batch", disable=(rank != 0))
        for batch_idx, (batch_chunks, collated) in enumerate(pbar):
            # Aggregator: batched forward over unique documents
            doc_indices = list(set(c["doc_idx"] for c in batch_chunks))
            cls_seqs = [doc_cls[d] for d in doc_indices]
            max_n = max(s.shape[0] for s in cls_seqs)
            padded = torch.zeros(len(doc_indices), max_n, cls_seqs[0].shape[1])
            agg_mask = torch.zeros(len(doc_indices), max_n)
            for i, s in enumerate(cls_seqs):
                padded[i, :s.shape[0]] = s
                agg_mask[i, :s.shape[0]] = 1

            padded_dev = padded.to(device)
            agg_mask_dev = agg_mask.to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                enriched = aggregator(padded_dev, agg_mask_dev)
                doc_to_cls = {d: enriched[i] for i, d in enumerate(doc_indices)}
                batch_cls = torch.stack(
                    [doc_to_cls[c["doc_idx"]] for c in batch_chunks])

                # Contrastive loss: second forward with different dropout mask
                loss_cl = torch.tensor(0.0, device=device)
                if training and contrastive_lambda > 0 and len(doc_indices) > 1:
                    enriched2 = aggregator(padded_dev, agg_mask_dev)
                    z1 = F.normalize(enriched.float(), dim=-1)
                    z2 = F.normalize(enriched2.float(), dim=-1)
                    loss_cl = compute_contrastive_loss(
                        z1, z2, temperature=contrastive_temp)

                # Load pre-collated decoder batch from pinned memory
                ids, num_mask, num_vals, attn_mask = (
                    t.to(device, non_blocking=True) for t in collated)

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

                # Build decoder embeddings and forward
                dec_embeds = model._build_embeds(
                    m_ids, m_nums, m_num_mask,
                    decoder_inner._get_tok_embeddings(),
                    decoder_inner.number_embedder,
                )
                text_logits, mag_logits = decoder(
                    dec_embeds, batch_cls, attn_mask)

            # Loss in fp32
            text_logits = text_logits.float()
            mag_logits = mag_logits.float()
            loss_recon, lt, lm = compute_loss(
                text_logits, mag_logits, labels_t, labels_m, config)
            loss = loss_recon + contrastive_lambda * loss_cl

            if training:
                accum_count += 1
                (loss / args.grad_accum_steps).backward()

                if accum_count % args.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(aggregator.parameters()) +
                        list(decoder.parameters()),
                        args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            # Metrics
            total_loss += loss.item()
            total_loss_text += lt.item()
            total_loss_mag += lm.item()
            cl_val = loss_cl.item() if torch.is_tensor(loss_cl) else 0.0
            total_loss_contrastive += cl_val
            recent_loss_text.append(lt.item())
            recent_loss_mag.append(lm.item())
            recent_loss_contrastive.append(cl_val)

            text_masked = labels_t.view(-1) != -100
            if text_masked.any():
                preds = text_logits.detach().view(
                    -1, config.vocab_size)[text_masked].argmax(-1)
                total_correct += (preds == labels_t.view(-1)[text_masked]).sum().item()
                total_masked += text_masked.sum().item()

            n_batches += 1

            if n_batches % 20 == 0:
                ma_lt = sum(recent_loss_text) / len(recent_loss_text)
                ma_lm = sum(recent_loss_mag) / len(recent_loss_mag)
                ma_cl = sum(recent_loss_contrastive) / len(recent_loss_contrastive)
                if training:
                    agg_lr = optimizer.param_groups[0]["lr"]
                    dec_lr = optimizer.param_groups[1]["lr"]
                    postfix = dict(
                        text=f"{ma_lt:.3f}", mag=f"{ma_lm:.3f}",
                        agg_lr=f"{agg_lr:.2e}", dec_lr=f"{dec_lr:.2e}")
                    if contrastive_lambda > 0:
                        postfix["cl"] = f"{ma_cl:.3f}"
                    pbar.set_postfix(**postfix)
                else:
                    pbar.set_postfix(
                        text=f"{ma_lt:.3f}", mag=f"{ma_lm:.3f}")

        # Flush remaining accumulated gradients
        if training and accum_count % args.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                list(aggregator.parameters()) +
                list(decoder.parameters()),
                args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    finally:
        torch.set_grad_enabled(prev_grad)
        random.setstate(rng_state)

    avg_loss = total_loss / max(n_batches, 1)
    avg_lt = total_loss_text / max(n_batches, 1)
    avg_lm = total_loss_mag / max(n_batches, 1)
    avg_cl = total_loss_contrastive / max(n_batches, 1)
    acc = total_correct / max(total_masked, 1)

    return {
        "loss": avg_loss,
        "loss_text": avg_lt,
        "loss_mag": avg_lm,
        "loss_contrastive": avg_cl,
        "text_acc": acc,
        "n_batches": n_batches,
    }


# ─── Dry Run (CPU-only, no model) ──────────────────────────────────────

def _dry_run(args):
    """Chunk + batch documents and report same-source pair stats, then exit."""
    random.seed(args.seed)

    tok_info = get_token_info(PRETRAINED_ID)
    cls_id = tok_info["cls_id"]
    sep_id = tok_info["sep_id"]
    nl_ids = tok_info["newline_ids"]

    print(f"Loading documents from {args.data}...")
    documents = torch.load(args.data, map_location="cpu", weights_only=False)
    train_docs, val_docs = split_documents(documents, args.val_ratio)
    del documents
    print(f"  {len(train_docs)} train, {len(val_docs)} val documents")

    for split_name, docs in [("Train", train_docs), ("Val", val_docs)]:
        print(f"\n--- {split_name} ---")

        all_enc_chunks = []
        all_dec_chunks = []
        doc_enc_spans = {}

        for doc_idx, doc in enumerate(docs):
            content_len = len(doc["input_ids"]) - 2
            if content_len < 1:
                continue
            boundaries = get_boundaries(doc["input_ids"], nl_ids)

            enc_target = random.randint(CHUNK_MIN, CHUNK_MAX)
            enc_spans = chunk_spans(content_len, boundaries, lambda: enc_target)
            doc_enc_spans[doc_idx] = enc_spans

            for ci, (s, e) in enumerate(enc_spans):
                chunk = extract_chunk(doc, s, e, cls_id, sep_id)
                chunk["doc_idx"] = doc_idx
                chunk["chunk_idx"] = ci
                all_enc_chunks.append(chunk)

            n_cls = len(enc_spans)
            dec_spans = chunk_spans(
                content_len, boundaries,
                lambda: random.randint(CHUNK_MIN, CHUNK_MAX))
            for ci, (s, e) in enumerate(dec_spans):
                chunk = extract_chunk(doc, s, e, cls_id, sep_id)
                chunk["doc_idx"] = doc_idx
                chunk["n_cls"] = n_cls
                all_dec_chunks.append(chunk)

        n_enc = len(all_enc_chunks)
        n_dec = len(all_dec_chunks)
        avg_enc = sum(c["seq_length"] for c in all_enc_chunks) / max(n_enc, 1)
        avg_dec = sum(c["seq_length"] for c in all_dec_chunks) / max(n_dec, 1)
        print(f"  Chunks: {n_enc} encoder (avg {avg_enc:.0f}), "
              f"{n_dec} decoder (avg {avg_dec:.0f})")

        avg_cls_per_doc = n_enc / max(len(doc_enc_spans), 1)
        batch_budget = (DEC_COST_WEIGHT * args.decoder_token_budget
                        + AGG_COST_WEIGHT * avg_cls_per_doc)
        print(f"  Batch budget: {batch_budget:.0f} "
              f"(avg {avg_cls_per_doc:.1f} CLS/doc)")
        dec_batches = form_batches_2d(all_dec_chunks, batch_budget)
        print(f"  {len(dec_batches)} decoder batches")

        # Padding waste
        total_dec_useful = total_dec_padded = 0
        total_agg_useful = total_agg_padded = 0
        for batch in dec_batches:
            max_dec = max(c["seq_length"] for c in batch)
            total_dec_useful += sum(c["seq_length"] for c in batch)
            total_dec_padded += max_dec * len(batch)
            batch_docs = {}
            for c in batch:
                batch_docs[c["doc_idx"]] = c["n_cls"]
            max_cls = max(batch_docs.values())
            total_agg_useful += sum(batch_docs.values())
            total_agg_padded += max_cls * len(batch_docs)
        dec_waste = 1.0 - total_dec_useful / max(total_dec_padded, 1)
        agg_waste = 1.0 - total_agg_useful / max(total_agg_padded, 1)
        weighted_waste = (
            (DEC_COST_WEIGHT * (total_dec_padded - total_dec_useful)
             + AGG_COST_WEIGHT * (total_agg_padded - total_agg_useful))
            / max(DEC_COST_WEIGHT * total_dec_padded
                  + AGG_COST_WEIGHT * total_agg_padded, 1)
        )
        print(f"  Padding waste: decoder {dec_waste:.1%}, "
              f"aggregator {agg_waste:.1%}, "
              f"weighted {weighted_waste:.1%}")

        # Contrastive positive stats
        avg_docs, min_docs, max_docs, n_cl, n_total = \
            evaluate_contrastive_positives(dec_batches)
        print(f"  Contrastive positives: {avg_docs:.1f} avg docs/batch "
              f"(min {min_docs}, max {max_docs}), "
              f"{n_cl}/{n_total} batches with ≥2 docs")


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train CLS Aggregator")
    parser.add_argument("--data", default="mlm_data/documents.pt")
    parser.add_argument("--checkpoint",
                        default="checkpoints/t5_cls/checkpoint_epoch5/full_model.pt")
    parser.add_argument("--output_dir", default="checkpoints/cls_aggregator")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decoder_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--encoder_token_budget", type=int, default=32768)
    parser.add_argument("--decoder_token_budget", type=int, default=16384)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--mask_prob_min", type=float, default=0.15)
    parser.add_argument("--mask_prob_max", type=float, default=0.85)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    # Aggregator hyperparams
    parser.add_argument("--agg_layers", type=int, default=6)
    parser.add_argument("--agg_heads", type=int, default=16)
    parser.add_argument("--agg_hidden", type=int, default=768)
    parser.add_argument("--agg_dropout", type=float, default=0.1)
    parser.add_argument("--contrastive_lambda", type=float, default=0.1,
                        help="Weight for contrastive loss (0 to disable)")
    parser.add_argument("--contrastive_temp", type=float, default=0.05,
                        help="InfoNCE temperature")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for kernel fusion")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only compute chunking/batching stats (no model, no GPU)")
    args = parser.parse_args()

    if args.dry_run:
        _dry_run(args)
        return

    rank, world_size = setup_distributed()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    rprint(rank, f"Device: {device} (world_size={world_size})")

    # Load T5 model (encoder frozen, decoder trainable)
    rprint(rank, f"\nLoading T5 model from {args.checkpoint}...")
    model, config = load_t5_model(args.checkpoint, device, rank=rank)

    decoder = model.decoder
    if args.compile:
        rprint(rank, "Compiling encoder, decoder, and aggregator with torch.compile...")
        model.encoder = torch.compile(model.encoder, dynamic=True)
        decoder = torch.compile(decoder, dynamic=True)

    # Token info
    tok_info = get_token_info(PRETRAINED_ID)

    # Load documents
    rprint(rank, f"\nLoading documents from {args.data}...")
    documents = torch.load(args.data, map_location="cpu", weights_only=False)
    train_docs, val_docs = split_documents(documents, args.val_ratio)
    del documents
    rprint(rank, f"  {len(train_docs)} train, {len(val_docs)} val documents")

    # Build aggregator
    aggregator = CLSAggregator(
        hidden_size=args.agg_hidden,
        num_heads=args.agg_heads,
        num_layers=args.agg_layers,
        dropout=args.agg_dropout,
    ).to(device)
    n_agg = sum(p.numel() for p in aggregator.parameters()) / 1e6
    rprint(rank, f"\nAggregator: {n_agg:.1f}M params")

    if args.compile:
        aggregator = torch.compile(aggregator, dynamic=True)

    # Wrap trainable modules in DDP
    if world_size > 1:
        aggregator = DDP(aggregator, device_ids=[rank])
        decoder = DDP(decoder, device_ids=[rank])

    # Optimizer & schedule (separate LR for decoder)
    optimizer = torch.optim.AdamW([
        {"params": aggregator.parameters(), "lr": args.lr},
        {"params": decoder.parameters(), "lr": args.decoder_lr},
    ], weight_decay=args.weight_decay)

    total_content = sum(len(d["input_ids"]) - 2 for d in train_docs)
    est_batches = total_content / args.decoder_token_budget
    total_steps = int(est_batches * args.epochs / args.grad_accum_steps)

    scheduler = get_cosine_schedule(optimizer, args.warmup_steps, total_steps)
    rprint(rank,
           f"Estimated ~{est_batches:.0f} batches/epoch, {total_steps} total steps")

    # Resume
    start_epoch = 0
    os.makedirs(args.output_dir, exist_ok=True)

    if args.resume:
        latest = os.path.join(args.output_dir, "latest.pt")
        if os.path.exists(latest):
            ckpt = torch.load(latest, map_location="cpu", weights_only=False)
            agg_unwrap = (aggregator.module if isinstance(aggregator, DDP)
                          else aggregator)
            agg_unwrap.load_state_dict(ckpt["aggregator"])
            if "decoder" in ckpt:
                model.decoder.load_state_dict(ckpt["decoder"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            rprint(rank, f"Resumed from epoch {start_epoch}")
            del ckpt

    # Training loop
    try:
        for epoch in range(start_epoch, args.epochs):
            rprint(rank, f"\n{'='*70}")
            rprint(rank, f"Epoch {epoch + 1}/{args.epochs}")
            rprint(rank, f"{'='*70}")

            # Train
            t0 = time.time()
            train_m = run_epoch(
                aggregator, decoder, model, train_docs, optimizer, scheduler,
                device, epoch, args, config, tok_info, training=True,
                rank=rank, world_size=world_size)
            t_train = time.time() - t0

            cl_str = (f", cl={train_m['loss_contrastive']:.4f}"
                      if args.contrastive_lambda > 0 else "")
            rprint(rank,
                   f"\n  Train: loss={train_m['loss']:.4f} "
                   f"(text={train_m['loss_text']:.4f}, "
                   f"mag={train_m['loss_mag']:.4f}{cl_str}) "
                   f"acc={train_m['text_acc']:.1%} [{t_train:.0f}s]")

            # Validate
            t0 = time.time()
            val_m = run_epoch(
                aggregator, decoder, model, val_docs, optimizer, scheduler,
                device, epoch, args, config, tok_info, training=False,
                rank=rank, world_size=world_size)
            t_val = time.time() - t0

            rprint(rank,
                   f"  Val:   loss={val_m['loss']:.4f} "
                   f"(text={val_m['loss_text']:.4f}, "
                   f"mag={val_m['loss_mag']:.4f}) "
                   f"acc={val_m['text_acc']:.1%} [{t_val:.0f}s]")

            # Save checkpoint (rank 0 only)
            if rank == 0:
                agg_unwrap = (aggregator.module if isinstance(aggregator, DDP)
                              else aggregator)
                ckpt = {
                    "aggregator": agg_unwrap.state_dict(),
                    "decoder": model.decoder.state_dict(),
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
                rprint(rank, f"  Saved checkpoint: epoch_{epoch + 1}.pt")

            # Sync all ranks before next epoch
            if world_size > 1:
                dist.barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
