"""
Evaluate CLS Aggregator embedding quality.

For every document:
1. Chunk → CLS-embed each chunk (encoder, frozen)
2. Aggregate chunk-level CLS tokens → single document embedding (aggregator)

Reports cosine similarity distribution, per-dimension variance, norms,
and effective dimensionality (PCA) across all document embeddings.

Usage (multi-GPU):
    torchrun --nproc_per_node=4 -m cls_aggregator_training_pipeline.eval_aggregator_embeddings

Usage (single GPU):
    python -m cls_aggregator_training_pipeline.eval_aggregator_embeddings
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cls_aggregator_training_pipeline.aggregator import CLSAggregator
from chunk_utils import (
    get_token_info,
    get_boundaries,
    chunk_spans,
    extract_chunk,
    pad_and_collate,
)
from financial_bert import FinancialModernBertConfig
from t5_style_training_pipeline.decoder import T5StyleModel

PRETRAINED_ID = "answerdotai/ModernBERT-base"
CHUNK_MIN = 128
CHUNK_MAX = 1024


# ─── Distributed ──────────────────────────────────────────────────────────

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


# ─── Batching (reused from train.py) ──────────────────────────────────────

def form_batches(items, token_budget):
    """Group length-sorted items into batches (max_len * batch_size <= budget)."""
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


# ─── CLS Computation ─────────────────────────────────────────────────────

@torch.no_grad()
def compute_all_cls(model, enc_chunks, device, token_budget, pad_id,
                    rank=0, world_size=1):
    """Compute CLS embeddings for all encoder chunks. Returns doc_idx -> (N, D)."""
    batches = form_batches(enc_chunks, token_budget)
    my_batches = batches[rank::world_size]

    hidden_size = None
    results = {}  # doc_idx -> {chunk_idx -> cls_vec}

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

    torch.cuda.empty_cache()

    # Multi-GPU merge via all-gather
    if world_size > 1:
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
        hidden_size = 768

    doc_cls = {}
    for doc_idx, chunks_dict in results.items():
        n = max(chunks_dict.keys()) + 1
        tensor = torch.zeros(n, hidden_size)
        for ci, vec in chunks_dict.items():
            tensor[ci] = vec
        doc_cls[doc_idx] = tensor

    return doc_cls


# ─── Aggregation ──────────────────────────────────────────────────────────

@torch.no_grad()
def aggregate_all(aggregator, doc_cls, device, batch_budget=4096, rank=0):
    """Run aggregator on all documents. Returns (n_docs, D) tensor on CPU.

    Batches documents by n_chunks to minimise padding: sorts by length,
    packs into batches where max_n_chunks * batch_size <= batch_budget.
    """
    items = [(doc_idx, cls_mat) for doc_idx, cls_mat in doc_cls.items()]
    items.sort(key=lambda x: x[1].shape[0])

    # Form batches
    batches = []
    current = []
    current_max = 0
    for item in items:
        n = item[1].shape[0]
        new_max = max(current_max, n)
        if current and new_max * (len(current) + 1) > batch_budget:
            batches.append(current)
            current = [item]
            current_max = n
        else:
            current.append(item)
            current_max = new_max
    if current:
        batches.append(current)

    all_embeds = []
    pbar = tqdm(batches, desc="  Aggregating", unit="batch", disable=(rank != 0))
    for batch in pbar:
        max_n = max(cls_mat.shape[0] for _, cls_mat in batch)
        B = len(batch)
        D = batch[0][1].shape[1]

        padded = torch.zeros(B, max_n, D)
        mask = torch.zeros(B, max_n, dtype=torch.long)
        for i, (_, cls_mat) in enumerate(batch):
            n = cls_mat.shape[0]
            padded[i, :n] = cls_mat
            mask[i, :n] = 1

        padded = padded.to(device)
        mask = mask.to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            doc_emb = aggregator(padded, attention_mask=mask)  # (B, D)

        all_embeds.append(doc_emb.float().cpu())

    return torch.cat(all_embeds, dim=0)


# ─── Analysis ─────────────────────────────────────────────────────────────

def analyse_embeddings(all_embs):
    """Print cosine similarity, variance, norms, PCA stats."""
    n, d = all_embs.shape
    print(f"\nEmbeddings: {n} documents, {d} dimensions\n")

    # Normalize
    norms = all_embs.norm(dim=-1, keepdim=True)
    normed = all_embs / norms

    # --- Cosine similarity ---
    print("=== Cosine Similarity ===")
    n_pairs = min(100_000, n * (n - 1) // 2)
    idx1 = torch.randint(0, n, (n_pairs,))
    idx2 = torch.randint(0, n, (n_pairs,))
    valid = idx1 != idx2
    idx1, idx2 = idx1[valid], idx2[valid]
    cos_sims = (normed[idx1] * normed[idx2]).sum(dim=-1)

    print(f"  Mean:   {cos_sims.mean().item():.4f}")
    print(f"  Std:    {cos_sims.std().item():.4f}")
    print(f"  Min:    {cos_sims.min().item():.4f}")
    print(f"  Max:    {cos_sims.max().item():.4f}")
    print(f"  Median: {cos_sims.median().item():.4f}")

    print("\n  Distribution:")
    for lo, hi in [(-1, 0), (0, 0.5), (0.5, 0.7), (0.7, 0.8),
                   (0.8, 0.9), (0.9, 0.95), (0.95, 1.0)]:
        frac = ((cos_sims >= lo) & (cos_sims < hi)).float().mean().item()
        print(f"    [{lo:.2f}, {hi:.2f}): {frac*100:.1f}%")

    # --- Per-dimension variance ---
    print("\n=== Per-dimension Variance ===")
    var_per_dim = all_embs.var(dim=0)
    print(f"  Mean variance:   {var_per_dim.mean().item():.6f}")
    print(f"  Std of variance: {var_per_dim.std().item():.6f}")
    print(f"  Min variance:    {var_per_dim.min().item():.6f}")
    print(f"  Max variance:    {var_per_dim.max().item():.6f}")
    print(f"  Total variance:  {var_per_dim.sum().item():.4f}")

    # --- Norms ---
    print("\n=== Embedding Norms ===")
    norms_flat = norms.squeeze()
    print(f"  Mean norm: {norms_flat.mean().item():.4f}")
    print(f"  Std norm:  {norms_flat.std().item():.4f}")
    print(f"  Min norm:  {norms_flat.min().item():.4f}")
    print(f"  Max norm:  {norms_flat.max().item():.4f}")

    # --- PCA / Effective Dimensionality ---
    print("\n=== PCA / Effective Dimensionality ===")
    centered = all_embs - all_embs.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    eigenvalues = (S ** 2) / (n - 1)
    explained_var_ratio = eigenvalues / eigenvalues.sum()
    cumsum = explained_var_ratio.cumsum(dim=0)

    d_eff = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
    print(f"  Effective dimensionality (participation ratio): {d_eff.item():.1f}")

    p = explained_var_ratio.clamp(min=1e-12)
    entropy = -(p * p.log()).sum()
    d_eff_entropy = entropy.exp()
    print(f"  Effective dimensionality (exp entropy):         {d_eff_entropy.item():.1f}")

    for k in [1, 5, 10, 25, 50, 100, 200]:
        if k <= len(cumsum):
            print(f"  Top-{k:>3d} components: {cumsum[k-1].item()*100:.1f}% variance")

    # --- Isotropy ---
    print(f"\n=== Isotropy ===")
    print(f"  Mean pairwise cosine similarity: {cos_sims.mean().item():.4f}")
    print(f"  (Ideal isotropic ~ 0.0, collapsed ~ 1.0)")

    print(f"\n  Top-10 singular values: {S[:10].tolist()}")
    print(f"  Ratio S[0]/S[9]: {S[0].item()/S[9].item():.2f}")


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CLS Aggregator embedding quality")
    parser.add_argument("--data", default="mlm_data/documents.pt")
    parser.add_argument("--t5_checkpoint",
                        default="checkpoints/t5_cls/checkpoint_epoch5/full_model.pt")
    parser.add_argument("--aggregator_checkpoint",
                        default="checkpoints/cls_aggregator/aggregator.pt")
    parser.add_argument("--encoder_token_budget", type=int, default=16384)
    parser.add_argument("--agg_batch_budget", type=int, default=4096,
                        help="max_n_chunks * batch_size for aggregator batching")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    rprint(rank, f"Device: {device}, world_size: {world_size}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ─── Load T5 model (encoder only needed) ──────────────────────────
    rprint(rank, "\n### Loading T5 model...")
    t5_ckpt = torch.load(args.t5_checkpoint, map_location="cpu", weights_only=False)
    config = FinancialModernBertConfig.from_pretrained(PRETRAINED_ID)
    config.num_magnitude_bins = t5_ckpt["args"].get("num_magnitude_bins", 128)

    model = T5StyleModel(config)
    state_dict = {k.removeprefix("_orig_mod."): v
                  for k, v in t5_ckpt["model_state_dict"].items()}
    del t5_ckpt
    model.load_state_dict(state_dict)
    del state_dict
    model.encoder.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)

    # ─── Load aggregator ──────────────────────────────────────────────
    rprint(rank, "### Loading aggregator...")
    aggregator = CLSAggregator()
    agg_sd = torch.load(args.aggregator_checkpoint, map_location="cpu",
                        weights_only=False)
    agg_sd = {k.removeprefix("_orig_mod."): v for k, v in agg_sd.items()}
    aggregator.load_state_dict(agg_sd)
    del agg_sd
    aggregator.eval()
    for p in aggregator.parameters():
        p.requires_grad_(False)
    aggregator.to(device)

    # ─── Load documents ───────────────────────────────────────────────
    rprint(rank, f"### Loading {args.data}...")
    documents = torch.load(args.data, map_location="cpu", weights_only=False)
    rprint(rank, f"  {len(documents)} documents")

    # ─── Chunk all documents ──────────────────────────────────────────
    rprint(rank, "### Chunking documents...")
    tok_info = get_token_info(PRETRAINED_ID)
    cls_id, sep_id, pad_id = tok_info["cls_id"], tok_info["sep_id"], tok_info["pad_id"]
    nl_ids = tok_info["newline_ids"]

    all_enc_chunks = []
    for doc_idx, doc in enumerate(documents):
        content_len = doc["seq_length"] - 2  # strip CLS/SEP
        if content_len < CHUNK_MIN:
            continue
        boundaries = get_boundaries(doc["input_ids"], nl_ids)
        enc_target = random.randint(CHUNK_MIN, CHUNK_MAX)
        spans = chunk_spans(content_len, boundaries, lambda: enc_target)

        for ci, (s, e) in enumerate(spans):
            chunk = extract_chunk(doc, s, e, cls_id, sep_id)
            chunk["doc_idx"] = doc_idx
            chunk["chunk_idx"] = ci
            all_enc_chunks.append(chunk)

    n_docs_with_chunks = len({c["doc_idx"] for c in all_enc_chunks})
    rprint(rank, f"  {len(all_enc_chunks)} chunks from {n_docs_with_chunks} documents")

    # ─── Step 1: CLS-embed all chunks ────────────────────────────────
    rprint(rank, "### Computing CLS embeddings...")
    all_enc_chunks.sort(key=lambda c: c["seq_length"])
    doc_cls = compute_all_cls(
        model, all_enc_chunks, device, args.encoder_token_budget, pad_id,
        rank=rank, world_size=world_size,
    )
    del all_enc_chunks, model
    torch.cuda.empty_cache()

    rprint(rank, f"  CLS embeddings for {len(doc_cls)} documents")

    # ─── Step 2: Aggregate ────────────────────────────────────────────
    rprint(rank, "### Running aggregator...")
    all_embs = aggregate_all(aggregator, doc_cls, device,
                             batch_budget=args.agg_batch_budget, rank=rank)
    del doc_cls, aggregator
    torch.cuda.empty_cache()

    # ─── Step 3: Analyse (rank 0 only) ───────────────────────────────
    if rank == 0:
        analyse_embeddings(all_embs)

    cleanup_distributed()


if __name__ == "__main__":
    main()
