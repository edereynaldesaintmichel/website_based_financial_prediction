"""
Evaluate word embedding quality from ModernBERT embedding dictionary.

Samples 1000 random word embeddings and reports cosine similarity statistics,
variance, and effective dimensionality (PCA) — for both base ModernBERT
and the fine-tuned FinancialModernBert checkpoint.
"""
import os
import sys
import random

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from financial_bert import build_model


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
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
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


def main():
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_words = 1000
    checkpoint_path = "checkpoints/mlm_full/checkpoint_epoch3/full_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Device: {device}")

    # Sample random token indices (avoid special tokens at the very start)
    # ModernBERT vocab size is ~50k; sample from reasonable range
    print(f"\nBuilding base model...")
    base_model = build_model("answerdotai/ModernBERT-base")
    vocab_size = base_model.config.vocab_size

    indices = random.sample(range(100, vocab_size), n_words)
    indices_tensor = torch.tensor(indices, dtype=torch.long)

    # --- Base model embeddings ---
    base_emb_layer = base_model._get_embedding_layer()
    with torch.no_grad():
        base_embeddings = base_emb_layer(indices_tensor).float()
    analyze_embeddings(base_embeddings, "Base ModernBERT word embeddings")

    # --- Fine-tuned model embeddings ---
    print(f"\nLoading fine-tuned checkpoint from {checkpoint_path}...")
    ft_model = build_model("answerdotai/ModernBERT-base")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    ft_model.load_state_dict(state_dict)

    ft_emb_layer = ft_model._get_embedding_layer()
    with torch.no_grad():
        ft_embeddings = ft_emb_layer(indices_tensor).float()
    analyze_embeddings(ft_embeddings, "Fine-tuned FinancialModernBert word embeddings")


if __name__ == "__main__":
    main()