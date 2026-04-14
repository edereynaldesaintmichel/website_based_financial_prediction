"""
Latent-MLM training for the CLS aggregator.

Pipeline:
1. Chunk every document with boundary-aware random chunk sizes.
2. Encode every chunk with the frozen T5 encoder to get per-chunk CLS embeddings.
   Results are cached on disk per epoch (expensive, done once per seed/epoch).
3. Per batch of documents:
      target = sg(W(cls))                          # frozen teacher in latent
      x_in   = target, with 20% of positions replaced by the learnable mask
               token (per-doc, min 2 masked positions, at least 1 visible)
      pred   = aggregator(x_in)                    # (B, N, D)
      L_mse  = MSE(pred[masked], target[masked])
      L_reg  = mean_k SIGReg({ pred[i, k-th masked pos of doc i] : i })
      loss   = L_mse + λ · L_reg
   W is trained only through the aggregator's input path (unmasked positions).

Inference: mean-pool `aggregator(cls_tokens)` over valid positions.

Usage:
    python -m cls_aggregator_training_pipeline.train \
        --data mlm_data/documents.pt \
        --checkpoint checkpoints/t5_cls/checkpoint_epoch5/full_model.pt

Multi-GPU:
    torchrun --nproc_per_node=4 -m cls_aggregator_training_pipeline.train ...
"""
import argparse
import math
import os
import random
import sys
import threading
import time
from collections import deque

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cls_aggregator_training_pipeline.aggregator import CLSAggregator
from cls_aggregator_training_pipeline.muon import Muon, CombinedOptimizer
from chunk_utils import (
    get_token_info, get_boundaries, chunk_spans, extract_chunk, pad_and_collate,
)
from financial_bert import FinancialModernBertConfig
from split_utils import split_documents
from t5_style_training_pipeline.decoder import T5StyleModel

PRETRAINED_ID = "answerdotai/ModernBERT-base"

HF_REPO_ID = "edereynal/financial_prediction"
HF_REPO_TYPE = "dataset"

CHUNK_MIN = 128
CHUNK_MAX = 1024
MIN_DOC_CHUNKS = 3          # need at least one masked + one visible, with headroom
MIN_VIEW_SIZE = 32          # SIGReg views with fewer samples are skipped


# ─── Distributed helpers ────────────────────────────────────────────────────

def setup_distributed():
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
    if rank == 0:
        print(*args, **kwargs)


# ─── HF upload (background, best-effort) ────────────────────────────────────

def upload_to_hf_async(local_path, repo_path):
    """Upload a single file to HF in a background thread. Never blocks or
    raises — failures are logged and training continues."""
    def _run():
        try:
            from huggingface_hub import HfApi
            HfApi().upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=HF_REPO_ID,
                repo_type=HF_REPO_TYPE,
            )
            print(f"  HF: uploaded {repo_path}")
        except Exception as e:
            print(f"  HF upload failed for {repo_path} (non-fatal): {e}")
    threading.Thread(target=_run, daemon=True).start()


# ─── T5 encoder (frozen) ────────────────────────────────────────────────────

def load_encoder(checkpoint_path, device, rank=0):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = FinancialModernBertConfig.from_pretrained(PRETRAINED_ID)
    config.num_magnitude_bins = ckpt["args"].get("num_magnitude_bins", 128)

    model = T5StyleModel(config)
    state_dict = {k.removeprefix("_orig_mod."): v
                  for k, v in ckpt["model_state_dict"].items()
                  if not k.removeprefix("_orig_mod.").startswith("decoder.")}
    del ckpt
    model.load_state_dict(state_dict, strict=False)
    del state_dict

    model.encoder.eval()
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    # Decoder is unused — drop it to free memory.
    del model.decoder
    torch.cuda.empty_cache()

    model.to(device)
    n_enc = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    rprint(rank, f"  Loaded T5 encoder: {n_enc:.1f}M params (frozen)")
    return model, config


# ─── Chunking & CLS encoding ────────────────────────────────────────────────

def form_encoder_batches(items, token_budget):
    batches, current, current_max = [], [], 0
    for item in items:
        l = item["seq_length"]
        new_max = max(current_max, l)
        if current and new_max * (len(current) + 1) > token_budget:
            batches.append(current)
            current, current_max = [item], l
        else:
            current.append(item)
            current_max = new_max
    if current:
        batches.append(current)
    return batches


@torch.no_grad()
def compute_all_cls(model, enc_chunks, device, token_budget, pad_id,
                    rank=0, world_size=1, cache_path=None):
    """Returns dict: doc_idx → Tensor(n_chunks, D) on CPU."""
    batches = form_encoder_batches(enc_chunks, token_budget)
    my_batches = batches[rank::world_size]

    hidden_size = None
    results = {}  # doc_idx → {chunk_idx: cls_vec}

    pbar = tqdm(my_batches, desc="  CLS embeddings", unit="batch",
                disable=(rank != 0))
    for batch in pbar:
        ids, num_mask, num_vals, attn_mask = pad_and_collate(batch, pad_id)
        ids = ids.to(device); num_mask = num_mask.to(device)
        num_vals = num_vals.to(device); attn_mask = attn_mask.to(device)

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
            d, c = chunk["doc_idx"], chunk["chunk_idx"]
            results.setdefault(d, {})[c] = cls_vecs[i]

    torch.cuda.empty_cache()

    # Merge across ranks via disk shards (avoids all-gather memory spike).
    if world_size > 1 and cache_path is not None:
        shard = cache_path + f".shard_{rank}"
        torch.save(results, shard)
        del results
        dist.barrier()
        if rank == 0:
            merged = {}
            for r in range(world_size):
                s = torch.load(cache_path + f".shard_{r}",
                               map_location="cpu", weights_only=False)
                for d, cd in s.items():
                    merged.setdefault(d, {}).update(cd)
                del s
            results = merged
        else:
            return None
    elif world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, results)
        merged = {}
        for r in gathered:
            for d, cd in r.items():
                merged.setdefault(d, {}).update(cd)
        results = merged

    if hidden_size is None:
        hidden_size = 768

    doc_cls = {}
    for d, cd in results.items():
        n = max(cd.keys()) + 1
        t = torch.zeros(n, hidden_size)
        for ci, v in cd.items():
            t[ci] = v
        doc_cls[d] = t
    return doc_cls


def chunk_documents(documents, tok_info, rank=0):
    cls_id, sep_id, nl_ids = tok_info["cls_id"], tok_info["sep_id"], tok_info["newline_ids"]
    all_enc_chunks = []
    doc_n_chunks = {}
    for doc_idx, doc in enumerate(documents):
        content_len = len(doc["input_ids"]) - 2
        if content_len < 1:
            continue
        boundaries = get_boundaries(doc["input_ids"], nl_ids)
        target = random.randint(CHUNK_MIN, CHUNK_MAX)
        spans = chunk_spans(content_len, boundaries, lambda t=target: t)
        if len(spans) < MIN_DOC_CHUNKS:
            continue
        doc_n_chunks[doc_idx] = len(spans)
        for ci, (s, e) in enumerate(spans):
            chunk = extract_chunk(doc, s, e, cls_id, sep_id)
            chunk["doc_idx"] = doc_idx
            chunk["chunk_idx"] = ci
            all_enc_chunks.append(chunk)
    n = len(all_enc_chunks)
    avg = sum(c["seq_length"] for c in all_enc_chunks) / max(n, 1)
    rprint(rank, f"  Chunks: {n} (avg {avg:.0f} tokens), "
                 f"{len(doc_n_chunks)} documents ≥{MIN_DOC_CHUNKS} chunks")
    return all_enc_chunks, doc_n_chunks


# ─── Batching documents ─────────────────────────────────────────────────────

def form_doc_batches(doc_ids_to_n, cls_token_budget):
    """Group doc_idx values into batches whose padded CLS tensor respects a
    cls-token budget ((B+pad) × max_n_cls ≤ budget).

    Sorts docs by n_chunks and greedily packs consecutive docs until the next
    one would exceed the budget; then shuffles batch order.
    """
    items = sorted(doc_ids_to_n.items(), key=lambda x: x[1])  # (doc_idx, n)

    batches = []
    cur = []
    cur_max = 0
    for d, n in items:
        new_max = max(cur_max, n)
        if cur and (len(cur) + 1) * new_max > cls_token_budget:
            batches.append(cur)
            cur = [d]
            cur_max = n
        else:
            cur.append(d)
            cur_max = new_max
    if cur:
        batches.append(cur)
    random.shuffle(batches)
    return batches


def pad_doc_batch(doc_cls, doc_ids):
    seqs = [doc_cls[d] for d in doc_ids]
    B = len(seqs)
    max_n = max(s.shape[0] for s in seqs)
    D = seqs[0].shape[1]
    padded = torch.zeros(B, max_n, D)
    valid = torch.zeros(B, max_n, dtype=torch.bool)
    for i, s in enumerate(seqs):
        padded[i, :s.shape[0]] = s
        valid[i, :s.shape[0]] = True
    return padded, valid


# ─── Latent MLM loss ────────────────────────────────────────────────────────

def sample_mask(valid, mask_prob, min_mask, generator=None):
    """Per-row random mask.

    For each row: mask ceil(mask_prob * n_valid) positions, clamped to
    [min_mask, n_valid-1] so at least one token remains visible.
    Returns (B, N) bool.
    """
    B, N = valid.shape
    device = valid.device
    n_valid = valid.sum(dim=1)                                    # (B,)
    n_mask = (mask_prob * n_valid.float()).ceil().long()
    n_mask = torch.clamp(n_mask, min=min_mask)
    n_mask = torch.minimum(n_mask, torch.clamp(n_valid - 1, min=0))  # keep ≥1 visible

    scores = torch.rand(B, N, device=device, generator=generator)
    scores = scores.masked_fill(~valid, float("-inf"))
    sorted_scores, _ = scores.sort(dim=1, descending=True)
    idx = (n_mask - 1).clamp(min=0).unsqueeze(1)
    thresh = sorted_scores.gather(1, idx)                         # (B, 1)
    is_masked = (scores >= thresh) & valid & (n_mask.unsqueeze(1) > 0)
    return is_masked


def sigreg_loss(x, sketch_dim=64):
    """Force ECF(x) ~ ECF(standard Gaussian) in a random sketch direction.

    x: (N, C). Returns scalar.
    """
    N, C = x.shape
    A = torch.randn(C, sketch_dim, device=x.device, dtype=x.dtype)
    A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-6)

    t = torch.linspace(-5, 5, 17, device=x.device, dtype=x.dtype)
    exp_f = torch.exp(-0.5 * t * t)                               # (T,)

    proj = x @ A                                                  # (N, sketch)
    args = proj.unsqueeze(2) * t.view(1, 1, -1)                   # (N, sketch, T)
    ecf = torch.exp(1j * args.float()).mean(dim=0)                # (sketch, T)

    diff_sq = (ecf - exp_f.float().unsqueeze(0)).abs().square()
    err = diff_sq * exp_f.float().unsqueeze(0)
    loss = torch.trapz(err, t.float(), dim=1) * N
    return loss.mean()


def per_slot_view_sigreg(pred, is_masked):
    """Average SIGReg across per-slot views of masked predictions.

    For each doc, randomly permute its masked positions; the k-th view contains
    the k-th masked position from every doc that has ≥ k+1 masked positions.
    Views with < MIN_VIEW_SIZE samples are skipped.
    """
    B, N, D = pred.shape
    device = pred.device

    # For each row, sort indices by a random key so masked positions come first
    # in a shuffled order. sort is stable but we use a random tiebreak.
    rand_key = torch.rand(B, N, device=device)
    # Masked positions get a large score so they sort to the front.
    score = is_masked.float() + rand_key * 0.5                    # masked ∈ [1, 1.5], unmasked ∈ [0, 0.5]
    order = score.argsort(dim=1, descending=True)                 # (B, N)
    # Gather predictions in that order: (B, N, D).
    ordered_pred = pred.gather(1, order.unsqueeze(-1).expand(-1, -1, D))
    # Row k of ordered_pred[:, k, :] is the k-th masked pred (or junk if row has fewer masks).
    n_mask_per_row = is_masked.sum(dim=1)                         # (B,)
    max_m = int(n_mask_per_row.max().item()) if B > 0 else 0

    total = pred.new_zeros(())
    n_views = 0
    total_points = 0
    for k in range(max_m):
        keep = n_mask_per_row > k                                 # (B,)
        n_keep = int(keep.sum().item())
        if n_keep < MIN_VIEW_SIZE:
            continue
        view = ordered_pred[keep, k, :]                           # (n, D)
        total = total + sigreg_loss(view)
        n_views += 1
        total_points += n_keep
    if n_views == 0:
        return pred.new_zeros(()), 0, 0.0
    return total / n_views, n_views, total_points / n_views


# ─── LR schedule ────────────────────────────────────────────────────────────

def get_cosine_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─── Epoch ──────────────────────────────────────────────────────────────────

def cache_path_for(args, cache_idx, training):
    tag = "train" if training else "val"
    return os.path.join(
        args.cls_cache_dir, f"cls_cache_{tag}_epoch{cache_idx}.pt")


def ensure_cls_cache(cache_idx, training, model, documents, args, tok_info,
                     device, rank, world_size):
    """Compute (if missing) and persist the CLS cache for a given cache_idx.

    Chunking is seeded by cache_idx (not epoch), so caches are deterministic
    and can be reused when epochs cycle back to the same index.
    """
    cls_cache_path = cache_path_for(args, cache_idx, training)
    if os.path.exists(cls_cache_path):
        return cls_cache_path

    rng_state = random.getstate()
    random.seed(args.seed + cache_idx + (0 if training else 10000))
    try:
        all_enc_chunks, _ = chunk_documents(documents, tok_info, rank=rank)
    finally:
        random.setstate(rng_state)
    all_enc_chunks.sort(key=lambda c: c["seq_length"])

    assert model is not None, "encoder required to compute CLSs (no cache)"
    doc_cls = compute_all_cls(
        model, all_enc_chunks, device, args.encoder_token_budget,
        tok_info["pad_id"], rank=rank, world_size=world_size,
        cache_path=cls_cache_path)
    del all_enc_chunks

    if rank == 0:
        torch.save(doc_cls, cls_cache_path)
        rprint(rank, f"  Cached CLS embeddings to {cls_cache_path}")
        upload_to_hf_async(
            cls_cache_path,
            f"aggregator/cls_cache/{os.path.basename(cls_cache_path)}",
        )
        for r in range(world_size):
            shard = cls_cache_path + f".shard_{r}"
            if os.path.exists(shard):
                os.remove(shard)
    if world_size > 1:
        dist.barrier()
    torch.cuda.empty_cache()
    return cls_cache_path


def run_epoch(aggregator, optimizer, scheduler,
              device, epoch, args, training=True,
              rank=0, world_size=1):
    prefix = "Train" if training else "Val"
    aggregator.train() if training else aggregator.eval()

    rng_state = random.getstate()
    random.seed(args.seed + epoch * 1337 + (0 if training else 10000))

    # ─── Load CLS cache (cycles every num_cached_epochs) ────────────────
    cache_idx = epoch % args.num_cached_epochs
    cls_cache_path = cache_path_for(args, cache_idx, training)
    assert os.path.exists(cls_cache_path), (
        f"CLS cache missing: {cls_cache_path}. Precomputation should run "
        "before training.")
    rprint(rank, f"  Loading CLS cache (epoch {epoch} → idx {cache_idx}): "
                 f"{cls_cache_path}")
    doc_cls = torch.load(cls_cache_path, map_location="cpu", weights_only=False)
    doc_n_chunks = {d: t.shape[0] for d, t in doc_cls.items()}

    # ─── Form doc batches, shard across ranks ───────────────────────────
    batches = form_doc_batches(doc_n_chunks, args.cls_token_budget)
    rprint(rank, f"  {len(batches)} doc batches "
                 f"(budget {args.cls_token_budget} CLS tokens)")
    batches = batches[rank::world_size]

    # ─── Iterate ────────────────────────────────────────────────────────
    totals = dict(loss=0.0, mse=0.0, reg=0.0, n=0, n_views=0, avg_pts=0.0)
    recent_mse = deque(maxlen=100)
    recent_reg = deque(maxlen=100)

    prev_grad = torch.is_grad_enabled()
    if not training:
        torch.set_grad_enabled(False)
    if training:
        optimizer.zero_grad()
    accum = 0

    try:
        pbar = tqdm(batches, desc=f"  {prefix}", unit="batch", disable=(rank != 0))
        for doc_ids in pbar:
            padded, valid = pad_doc_batch(doc_cls, doc_ids)
            padded = padded.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)

            is_masked = sample_mask(
                valid, mask_prob=args.mask_prob, min_mask=args.min_mask)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred, target = aggregator(padded, valid, is_masked)

            pred_f = pred.float()
            target_f = target.float()

            # MSE on masked positions only.
            masked_flat = is_masked.view(-1)
            if masked_flat.any():
                mse = F.mse_loss(
                    pred_f.reshape(-1, pred_f.shape[-1])[masked_flat],
                    target_f.reshape(-1, target_f.shape[-1])[masked_flat],
                )
            else:
                mse = pred_f.new_zeros(())

            reg, n_views, avg_pts = per_slot_view_sigreg(pred_f, is_masked)
            loss = mse + args.sigreg_lambda * reg

            if training:
                accum += 1
                (loss / args.grad_accum_steps).backward()
                if accum % args.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        aggregator.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            totals["loss"] += loss.item()
            totals["mse"] += mse.item()
            totals["reg"] += reg.item() if torch.is_tensor(reg) else float(reg)
            totals["n"] += 1
            totals["n_views"] += n_views
            totals["avg_pts"] += avg_pts
            recent_mse.append(mse.item())
            recent_reg.append(reg.item() if torch.is_tensor(reg) else float(reg))

            if totals["n"] % 20 == 0:
                ma_mse = sum(recent_mse) / len(recent_mse)
                ma_reg = sum(recent_reg) / len(recent_reg)
                if training:
                    pbar.set_postfix(
                        mse=f"{ma_mse:.4f}", reg=f"{ma_reg:.4f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.2e}")
                else:
                    pbar.set_postfix(mse=f"{ma_mse:.4f}", reg=f"{ma_reg:.4f}")

        if training and accum % args.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                aggregator.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    finally:
        torch.set_grad_enabled(prev_grad)
        random.setstate(rng_state)

    n = max(totals["n"], 1)
    return {
        "loss": totals["loss"] / n,
        "mse": totals["mse"] / n,
        "reg": totals["reg"] / n,
        "avg_views_per_batch": totals["n_views"] / n,
        "avg_points_per_view": totals["avg_pts"] / n,
        "n_batches": totals["n"],
    }


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Latent-MLM CLS aggregator training")
    parser.add_argument("--data", default="mlm_data/documents.pt")
    parser.add_argument("--checkpoint",
                        default="checkpoints/t5_cls/checkpoint_epoch5/full_model.pt")
    parser.add_argument("--output_dir", default="checkpoints/cls_aggregator_latentmlm")
    parser.add_argument("--cls_cache_dir", default=None,
                        help="Where to read/write CLS caches. "
                             "Defaults to {output_dir}/cls_cache.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num_cached_epochs", type=int, default=5,
                        help="Number of distinct CLS caches to precompute. "
                             "Epoch e uses cache (e %% num_cached_epochs), so "
                             "epochs cycle through the precomputed shards.")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="AdamW LR for 1D params (norms, mask_token).")
    parser.add_argument("--muon_lr", type=float, default=0.02,
                        help="Muon LR for 2D hidden matrices.")
    parser.add_argument("--proj_lr", type=float, default=None,
                        help="Muon LR for the input projection W. Defaults to "
                             "muon_lr / 100 to damp teacher drift.")
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--encoder_token_budget", type=int, default=32768)
    parser.add_argument("--cls_token_budget", type=int, default=16384,
                        help="Per-batch cap on padded CLS tokens (B × max_n_cls).")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--mask_prob", type=float, default=0.2)
    parser.add_argument("--min_mask", type=int, default=2)
    parser.add_argument("--sigreg_lambda", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--agg_layers", type=int, default=12)
    parser.add_argument("--agg_heads", type=int, default=16)
    parser.add_argument("--agg_hidden", type=int, default=768)
    parser.add_argument("--agg_dropout", type=float, default=0.0)
    parser.add_argument("--freeze_projection", action="store_true",
                        help="Freeze the aggregator input projection W at identity "
                             "(teacher = raw CLS). Useful as a λ=0 diagnostic.")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    rank, world_size = setup_distributed()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    rprint(rank, f"Device: {device} (world_size={world_size})")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.cls_cache_dir is None:
        args.cls_cache_dir = os.path.join(args.output_dir, "cls_cache")
    os.makedirs(args.cls_cache_dir, exist_ok=True)

    # Defer loading the T5 encoder: needed only if the CLS cache is missing.
    model = None

    def ensure_encoder():
        nonlocal model
        if model is None:
            rprint(rank, f"\nLoading T5 encoder from {args.checkpoint}...")
            model, _ = load_encoder(args.checkpoint, device, rank=rank)
        return model

    tok_info = get_token_info(PRETRAINED_ID)

    rprint(rank, f"\nLoading documents from {args.data}...")
    documents = torch.load(args.data, map_location="cpu", weights_only=False)
    train_docs, val_docs = split_documents(documents, args.val_ratio)
    del documents
    rprint(rank, f"  {len(train_docs)} train, {len(val_docs)} val documents")

    aggregator = CLSAggregator(
        hidden_size=args.agg_hidden,
        num_heads=args.agg_heads,
        num_layers=args.agg_layers,
        dropout=args.agg_dropout,
    ).to(device)
    if args.freeze_projection:
        aggregator.W.weight.requires_grad_(False)
        rprint(rank, "  Projection W frozen at identity (teacher = raw CLS).")
    n_agg = sum(p.numel() for p in aggregator.parameters()) / 1e6
    rprint(rank, f"\nAggregator: {n_agg:.1f}M params")

    if args.compile:
        rprint(rank, "Compiling aggregator with torch.compile...")
        aggregator = torch.compile(aggregator, dynamic=True)

    if world_size > 1:
        aggregator = DDP(aggregator, device_ids=[rank])

    # Muon for 2D hidden matrices (Wqkv, Wo, w_gate/up/down). The input
    # projection W gets its own Muon group at a much lower LR to damp
    # teacher drift (W produces both target (sg) and input).
    # AdamW for the rest (LayerNorm params, mask_token).
    agg_unwrap = aggregator.module if isinstance(aggregator, DDP) else aggregator
    proj_param_ids = {id(agg_unwrap.W.weight)}
    muon_params = [p for p in aggregator.parameters()
                   if p.requires_grad and p.ndim == 2
                   and id(p) not in proj_param_ids]
    proj_params = [p for p in aggregator.parameters()
                   if p.requires_grad and id(p) in proj_param_ids]
    adamw_params = [p for p in aggregator.parameters()
                    if p.requires_grad and p.ndim != 2]
    proj_lr = args.proj_lr if args.proj_lr is not None else args.muon_lr / 100
    rprint(rank,
           f"Optimizer: Muon ({sum(p.numel() for p in muon_params)/1e6:.1f}M params) "
           f"+ Muon-proj ({sum(p.numel() for p in proj_params)/1e6:.2f}M params @ lr={proj_lr:.2e}) "
           f"+ AdamW ({sum(p.numel() for p in adamw_params)/1e6:.2f}M params)")
    muon_optims = [Muon(muon_params, lr=args.muon_lr, momentum=0.95,
                        weight_decay=args.weight_decay)]
    if proj_params:
        muon_optims.append(Muon(proj_params, lr=proj_lr, momentum=0.95,
                                weight_decay=args.weight_decay))
    optimizer = CombinedOptimizer(muon_optims + [
        torch.optim.AdamW(adamw_params, lr=args.lr,
                          weight_decay=args.weight_decay),
    ])

    # Rough step estimate for cosine schedule.
    avg_chunks = max(sum((len(d["input_ids"]) - 2) / ((CHUNK_MIN + CHUNK_MAX) / 2)
                         for d in train_docs), 1)
    est_batches = max(avg_chunks / args.cls_token_budget * 1.5, 1)
    total_steps = int(est_batches * args.epochs / args.grad_accum_steps) + args.warmup_steps
    scheduler = get_cosine_schedule(optimizer, args.warmup_steps, total_steps)
    rprint(rank, f"Estimated ~{est_batches:.0f} batches/epoch, "
                 f"{total_steps} total steps")

    start_epoch = 0
    resume_best_val = float("inf")
    if args.resume:
        best = os.path.join(args.output_dir, "best.pt")
        if os.path.exists(best):
            ckpt = torch.load(best, map_location="cpu", weights_only=False)
            agg_unwrap = aggregator.module if isinstance(aggregator, DDP) else aggregator
            agg_unwrap.load_state_dict(ckpt["aggregator"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            resume_best_val = ckpt.get("val_metrics", {}).get("loss", float("inf"))
            rprint(rank, f"Resumed from epoch {start_epoch} "
                         f"(best val loss so far: {resume_best_val:.4f})")
            del ckpt

    # ─── Precompute all CLS caches upfront (parallel across ranks) ─────
    K = args.num_cached_epochs
    rprint(rank, f"\nPrecomputing CLS caches (K={K}, train+val)...")
    any_missing = False
    for k in range(K):
        for tr_flag in (True, False):
            if not os.path.exists(cache_path_for(args, k, tr_flag)):
                any_missing = True
                break
        if any_missing:
            break
    if any_missing:
        ensure_encoder()
    for k in range(K):
        for tr_flag, docs in ((True, train_docs), (False, val_docs)):
            tag = "train" if tr_flag else "val"
            rprint(rank, f"  [cache {k+1}/{K}, {tag}]")
            ensure_cls_cache(k, tr_flag, model, docs, args, tok_info,
                             device, rank, world_size)
    # Encoder no longer needed — free its memory for the rest of training.
    if model is not None:
        del model
        model = None
        torch.cuda.empty_cache()
        rprint(rank, "  T5 encoder freed; entering training loop.")

    best_val_loss = resume_best_val
    try:
        for epoch in range(start_epoch, args.epochs):
            rprint(rank, f"\n{'='*70}\nEpoch {epoch + 1}/{args.epochs} "
                         f"(cache idx {epoch % K})\n{'='*70}")

            t0 = time.time()
            tr = run_epoch(aggregator, optimizer, scheduler,
                           device, epoch, args, training=True,
                           rank=rank, world_size=world_size)
            t_train = time.time() - t0
            rprint(rank, f"\n  Train: loss={tr['loss']:.4f} "
                         f"mse={tr['mse']:.4f} reg={tr['reg']:.4f} "
                         f"views/batch={tr['avg_views_per_batch']:.1f} "
                         f"pts/view={tr['avg_points_per_view']:.1f} "
                         f"[{t_train:.0f}s]")

            t0 = time.time()
            va = run_epoch(aggregator, optimizer, scheduler,
                           device, epoch, args, training=False,
                           rank=rank, world_size=world_size)
            t_val = time.time() - t0
            rprint(rank, f"  Val:   loss={va['loss']:.4f} "
                         f"mse={va['mse']:.4f} reg={va['reg']:.4f} "
                         f"pts/view={va['avg_points_per_view']:.1f} "
                         f"[{t_val:.0f}s]")

            if rank == 0:
                agg_unwrap = aggregator.module if isinstance(aggregator, DDP) else aggregator
                ckpt = {
                    "aggregator": agg_unwrap.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "train_metrics": tr,
                    "val_metrics": va,
                    "args": vars(args),
                }
                if va["loss"] < best_val_loss:
                    best_val_loss = va["loss"]
                    torch.save(ckpt, os.path.join(args.output_dir, "best.pt"))
                    rprint(rank, f"  New best (val loss={va['loss']:.4f}); "
                                 f"saved best.pt")
                else:
                    rprint(rank, f"  Val loss {va['loss']:.4f} ≥ best "
                                 f"{best_val_loss:.4f}; checkpoint not saved.")

            if world_size > 1:
                dist.barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
