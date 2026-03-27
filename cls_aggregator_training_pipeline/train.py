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

Usage (single GPU):
    python -m cls_aggregator_training_pipeline.train \
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
from collections import defaultdict, deque
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cls_aggregator_training_pipeline.aggregator import CLSAggregator
from financial_bert import FinancialBertTokenizer, FinancialModernBertConfig
from split_utils import split_documents
from t5_style_training_pipeline.decoder import T5StyleModel
from t5_style_training_pipeline.train import create_masked_inputs

PRETRAINED_ID = "answerdotai/ModernBERT-base"


CHUNK_MIN = 128
CHUNK_MAX = 1024
MIN_TAIL_CHUNK = 16  # absorb trailing chunks shorter than this


# ─── Model Loading ──────────────────────────────────────────────────────────

def load_t5_model(checkpoint_path, device):
    """Load T5 model from checkpoint. Encoder is frozen, decoder is trainable."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = FinancialModernBertConfig.from_pretrained(PRETRAINED_ID)
    config.num_magnitude_bins = ckpt["args"].get("num_magnitude_bins", 128)

    model = T5StyleModel(config, cross_attn_type="expanded_memory")
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
    print(f"  Loaded T5 model: encoder {n_enc:.1f}M (frozen), "
          f"decoder {n_dec:.1f}M (trainable)")
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
    """Compute CLS embeddings for all encoder chunks.

    Returns: dict of doc_idx → Tensor(N, D) on CPU.
    """
    batches = form_batches(enc_chunks, token_budget)

    hidden_size = None
    results = {}  # flat_idx → cls_vec

    pbar = tqdm(batches, desc="  CLS embeddings", unit="batch")
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
              device, epoch, args, config, tok_info, training=True):
    """Run one training or validation epoch."""
    prefix = "Train" if training else "Val"
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
    print(f"  Chunks: {n_enc} encoder (avg {avg_enc:.0f}), "
          f"{n_dec} decoder (avg {avg_dec:.0f})")

    # ─── Step 2: Compute CLS embeddings (cached to disk) ────────────────
    cache_tag = "train" if training else "val"
    cls_cache_path = os.path.join(
        args.output_dir, f"cls_cache_{cache_tag}_epoch{epoch}.pt")

    if os.path.exists(cls_cache_path):
        print(f"  Loading cached CLS embeddings from {cls_cache_path}")
        doc_cls = torch.load(cls_cache_path, map_location="cpu", weights_only=False)
        del all_enc_chunks
    else:
        all_enc_chunks.sort(key=lambda c: c["seq_length"])
        doc_cls = compute_all_cls(
            model, all_enc_chunks, device, args.encoder_token_budget, pad_id)
        del all_enc_chunks
        torch.save(doc_cls, cls_cache_path)
        print(f"  Cached CLS embeddings to {cls_cache_path}")
    torch.cuda.empty_cache()

    # ─── Step 3: 2D bucketing (dec_len × n_cls), form batches ───────────
    avg_cls_per_doc = n_enc / max(len(doc_enc_spans), 1)
    batch_budget = (DEC_COST_WEIGHT * args.decoder_token_budget
                    + AGG_COST_WEIGHT * avg_cls_per_doc)
    print(f"  Batch budget: {batch_budget:.0f} "
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
    print(f"  Padding waste: decoder {dec_waste:.1%}, "
          f"aggregator {agg_waste:.1%}, "
          f"weighted {weighted_waste:.1%}")
    print(f"  {len(dec_batches)} decoder batches")

    # ─── Step 4: Train/eval on decoder batches ──────────────────────────
    total_loss = 0.0
    total_loss_text = 0.0
    total_loss_mag = 0.0
    total_correct = 0
    total_masked = 0
    n_batches = 0
    accum_count = 0
    recent_loss_text = deque(maxlen=100)
    recent_loss_mag = deque(maxlen=100)

    prev_grad = torch.is_grad_enabled()
    if not training:
        torch.set_grad_enabled(False)

    if training:
        optimizer.zero_grad()

    try:
        pbar = tqdm(dec_batches, desc=f"  {prefix}", unit="batch")
        for batch_idx, batch_chunks in enumerate(pbar):
            # Aggregator: batched forward over unique documents
            doc_indices = list(set(c["doc_idx"] for c in batch_chunks))
            cls_seqs = [doc_cls[d] for d in doc_indices]
            max_n = max(s.shape[0] for s in cls_seqs)
            padded = torch.zeros(len(doc_indices), max_n, cls_seqs[0].shape[1])
            agg_mask = torch.zeros(len(doc_indices), max_n)
            for i, s in enumerate(cls_seqs):
                padded[i, :s.shape[0]] = s
                agg_mask[i, :s.shape[0]] = 1

            with torch.autocast("cuda", dtype=torch.bfloat16):
                enriched = aggregator(padded.to(device), agg_mask.to(device))
                doc_to_cls = {d: enriched[i] for i, d in enumerate(doc_indices)}
                batch_cls = torch.stack(
                    [doc_to_cls[c["doc_idx"]] for c in batch_chunks])

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

                # Build decoder embeddings and forward
                dec_embeds = model._build_embeds(
                    m_ids, m_nums, m_num_mask,
                    decoder._get_tok_embeddings(),
                    decoder.number_embedder,
                )
                text_logits, mag_logits = decoder(
                    dec_embeds, batch_cls, attn_mask)

            # Loss in fp32
            text_logits = text_logits.float()
            mag_logits = mag_logits.float()
            loss, lt, lm = compute_loss(
                text_logits, mag_logits, labels_t, labels_m, config)

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
            recent_loss_text.append(lt.item())
            recent_loss_mag.append(lm.item())

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
                if training:
                    agg_lr = optimizer.param_groups[0]["lr"]
                    dec_lr = optimizer.param_groups[1]["lr"]
                    pbar.set_postfix(
                        text=f"{ma_lt:.3f}", mag=f"{ma_lm:.3f}",
                        agg_lr=f"{agg_lr:.2e}", dec_lr=f"{dec_lr:.2e}")
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
    parser.add_argument("--decoder_lr", type=float, default=1e-5)
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
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for kernel fusion")
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.cuda.set_device(device)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    print(f"Device: {device}")

    # Load T5 model (encoder frozen, decoder trainable)
    print(f"\nLoading T5 model from {args.checkpoint}...")
    model, config = load_t5_model(args.checkpoint, device)

    decoder = model.decoder
    if args.compile:
        print("Compiling encoder and decoder with torch.compile...")
        model.encoder = torch.compile(model.encoder, dynamic=True)
        decoder = torch.compile(decoder, dynamic=True)

    # Token info
    tok_info = get_token_info(PRETRAINED_ID)

    # Load documents
    print(f"\nLoading documents from {args.data}...")
    documents = torch.load(args.data, map_location="cpu", weights_only=False)
    train_docs, val_docs = split_documents(documents, args.val_ratio)
    del documents
    print(f"  {len(train_docs)} train, {len(val_docs)} val documents")

    # Build aggregator
    aggregator = CLSAggregator(
        hidden_size=args.agg_hidden,
        num_heads=args.agg_heads,
        num_layers=args.agg_layers,
        dropout=args.agg_dropout,
    ).to(device)
    n_agg = sum(p.numel() for p in aggregator.parameters()) / 1e6
    print(f"\nAggregator: {n_agg:.1f}M params")

    # Optimizer & schedule (separate LR for decoder)
    optimizer = torch.optim.AdamW([
        {"params": aggregator.parameters(), "lr": args.lr},
        {"params": decoder.parameters(), "lr": args.decoder_lr},
    ], weight_decay=args.weight_decay)

    total_content = sum(len(d["input_ids"]) - 2 for d in train_docs)
    est_batches = total_content / args.decoder_token_budget
    total_steps = int(est_batches * args.epochs / args.grad_accum_steps)

    scheduler = get_cosine_schedule(optimizer, args.warmup_steps, total_steps)
    print(f"Estimated ~{est_batches:.0f} batches/epoch, {total_steps} total steps")

    # Resume
    start_epoch = 0
    os.makedirs(args.output_dir, exist_ok=True)

    if args.resume:
        latest = os.path.join(args.output_dir, "latest.pt")
        if os.path.exists(latest):
            ckpt = torch.load(latest, map_location="cpu", weights_only=False)
            aggregator.load_state_dict(ckpt["aggregator"])
            if "decoder" in ckpt:
                model.decoder.load_state_dict(ckpt["decoder"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            print(f"Resumed from epoch {start_epoch}")
            del ckpt

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*70}")

        # Train
        t0 = time.time()
        train_m = run_epoch(
            aggregator, decoder, model, train_docs, optimizer, scheduler,
            device, epoch, args, config, tok_info, training=True)
        t_train = time.time() - t0

        print(f"\n  Train: loss={train_m['loss']:.4f} "
              f"(text={train_m['loss_text']:.4f}, mag={train_m['loss_mag']:.4f}) "
              f"acc={train_m['text_acc']:.1%} [{t_train:.0f}s]")

        # Validate
        t0 = time.time()
        val_m = run_epoch(
            aggregator, decoder, model, val_docs, optimizer, scheduler,
            device, epoch, args, config, tok_info, training=False)
        t_val = time.time() - t0

        print(f"  Val:   loss={val_m['loss']:.4f} "
              f"(text={val_m['loss_text']:.4f}, mag={val_m['loss_mag']:.4f}) "
              f"acc={val_m['text_acc']:.1%} [{t_val:.0f}s]")

        # Save checkpoint
        ckpt = {
            "aggregator": aggregator.state_dict(),
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
        print(f"  Saved checkpoint: epoch_{epoch + 1}.pt")


if __name__ == "__main__":
    main()
