"""
Evaluate CLS embedding quality from FinancialModernBert encoder.

Samples random 512-token chunks from documents.pt, extracts CLS embeddings,
applies ZCA whitening as post-hoc regularization, and reports cosine similarity
statistics, variance, and effective dimensionality (PCA) for both raw and
regularized embeddings.
"""
import os
import sys
import random

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from financial_bert import build_model, FinancialModernBertConfig


def analyze_embeddings(embeddings: torch.Tensor, label: str):
    """Run full analysis on a [N, D] embedding matrix."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    norms = embeddings.norm(dim=-1, keepdim=True)
    normed = embeddings / norms

    # --- Cosine similarity stats ---
    print("\n=== Cosine Similarity ===")
    n_pairs = 50000
    idx1 = torch.randint(0, len(embeddings), (n_pairs,))
    idx2 = torch.randint(0, len(embeddings), (n_pairs,))
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]
    cos_sims = (normed[idx1] * normed[idx2]).sum(dim=-1)

    print(f"  Mean:   {cos_sims.mean().item():.4f}")
    print(f"  Std:    {cos_sims.std().item():.4f}")
    print(f"  Min:    {cos_sims.min().item():.4f}")
    print(f"  Max:    {cos_sims.max().item():.4f}")
    print(f"  Median: {cos_sims.median().item():.4f}")

    print("\n  Distribution:")
    for lo, hi in [(-1, 0), (0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.0)]:
        frac = ((cos_sims >= lo) & (cos_sims < hi)).float().mean().item()
        print(f"    [{lo:.2f}, {hi:.2f}): {frac*100:.1f}%")

    # --- Variance per dimension ---
    print("\n=== Per-dimension Variance ===")
    var_per_dim = embeddings.var(dim=0)
    print(f"  Mean variance:   {var_per_dim.mean().item():.6f}")
    print(f"  Std of variance: {var_per_dim.std().item():.6f}")
    print(f"  Min variance:    {var_per_dim.min().item():.6f}")
    print(f"  Max variance:    {var_per_dim.max().item():.6f}")
    print(f"  Total variance:  {var_per_dim.sum().item():.4f}")

    # --- Norm stats ---
    print("\n=== Embedding Norms ===")
    norms_flat = norms.squeeze()
    print(f"  Mean norm: {norms_flat.mean().item():.4f}")
    print(f"  Std norm:  {norms_flat.std().item():.4f}")
    print(f"  Min norm:  {norms_flat.min().item():.4f}")
    print(f"  Max norm:  {norms_flat.max().item():.4f}")

    # --- PCA / Effective Dimensionality ---
    print("\n=== PCA / Effective Dimensionality ===")
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    _, S, _ = torch.linalg.svd(centered, full_matrices=False)
    eigenvalues = (S ** 2) / (len(embeddings) - 1)
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

    print(f"\n=== Isotropy ===")
    print(f"  Mean pairwise cosine similarity: {cos_sims.mean().item():.4f}")
    print(f"  (Ideal isotropic ≈ 0.0, collapsed ≈ 1.0)")

    print(f"\n  Top-10 singular values: {S[:10].tolist()}")
    print(f"  Ratio S[0]/S[9]: {S[0].item()/S[9].item():.2f}")


def zca_whiten(embeddings: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Apply ZCA whitening: center, decorrelate, equalize variance.

    ZCA (Mahalanobis) whitening preserves the original axis alignment
    better than PCA whitening, so the result stays interpretable in
    the same coordinate frame.

    Returns whitened embeddings with identity covariance.
    """
    mean = embeddings.mean(dim=0, keepdim=True)
    centered = embeddings - mean

    cov = (centered.T @ centered) / (centered.shape[0] - 1)  # [D, D]

    # Eigen-decompose the covariance
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)  # ascending order
    # Whitening matrix: V @ diag(1/sqrt(λ)) @ V^T
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(eigenvalues.clamp(min=eps)))
    W = eigenvectors @ D_inv_sqrt @ eigenvectors.T

    return centered @ W


def main():
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_chunks = 1000
    chunk_size = 512
    checkpoint_path = "checkpoints/t5_cls/checkpoint_epoch5/encoder.pt"
    documents_path = "mlm_data/documents.pt"

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Device: {device}")

    # Load model
    print("Building model...")
    model = build_model("answerdotai/ModernBERT-base")

    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Load documents
    print(f"Loading {documents_path}...")
    documents = torch.load(documents_path, map_location="cpu", weights_only=False)
    print(f"  {len(documents)} documents")

    # Sample random chunks
    print(f"Sampling {n_chunks} random chunks of {chunk_size} tokens...")
    chunks = []
    attempts = 0
    while len(chunks) < n_chunks and attempts < n_chunks * 10:
        doc = random.choice(documents)
        seq_len = doc["seq_length"]
        if seq_len < chunk_size:
            attempts += 1
            continue
        start = random.randint(0, seq_len - chunk_size)
        chunks.append({
            "input_ids": doc["input_ids"][start:start + chunk_size],
            "is_number_mask": doc["is_number_mask"][start:start + chunk_size],
            "number_values": doc["number_values"][start:start + chunk_size],
        })
        attempts += 1

    print(f"  Collected {len(chunks)} chunks")

    # Extract CLS embeddings
    print("Extracting CLS embeddings...")
    all_cls = []
    batch_size = 4

    with torch.no_grad():
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            input_ids = torch.stack([c["input_ids"] for c in batch]).to(device)
            is_number_mask = torch.stack([c["is_number_mask"] for c in batch]).float().to(device)
            number_values = torch.stack([c["number_values"] for c in batch]).float().to(device)
            attention_mask = torch.ones_like(input_ids)

            text_embeds = model._get_embedding_layer()(input_ids)
            num_embeds = model.number_embedder(number_values)
            hidden_dim = model.config.hidden_size
            mask_expanded = is_number_mask.unsqueeze(-1).expand(-1, -1, hidden_dim).bool()
            final_inputs_embeds = torch.where(mask_expanded, num_embeds, text_embeds)

            backbone_out = model.modernbert(
                inputs_embeds=final_inputs_embeds,
                attention_mask=attention_mask,
            )
            cls_emb = backbone_out.last_hidden_state[:, 0, :]  # [B, hidden_dim]
            all_cls.append(cls_emb.cpu())

            if (i // batch_size) % 10 == 0:
                print(f"  {i + len(batch)}/{len(chunks)}")

    all_cls = torch.cat(all_cls, dim=0).float()  # [n_chunks, hidden_dim]
    print(f"  CLS embeddings shape: {all_cls.shape}")

    # Analyze raw CLS
    analyze_embeddings(all_cls, "Raw CLS embeddings")

    # ZCA whiten and analyze
    print("\nApplying ZCA whitening...")
    cls_whitened = zca_whiten(all_cls)
    analyze_embeddings(cls_whitened, "ZCA-whitened CLS embeddings")


if __name__ == "__main__":
    main()
