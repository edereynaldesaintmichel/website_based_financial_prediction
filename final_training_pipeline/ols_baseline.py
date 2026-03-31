"""
OLS baseline: frozen aggregator embeddings → ridge regression.

Loads pre-computed CLS caches, runs the frozen aggregator to produce
document embeddings, then fits closed-form ridge regression.

Usage:
    python -m final_training_pipeline.ols_baseline \
        --cache-dir /workspace/checkpoints/growth_predictor \
        --aggregator-checkpoint /workspace/data/aggregator_checkpoint/aggregator.pt \
        --epoch 0
"""
import argparse
import os
import sys
import time

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cls_aggregator_training_pipeline.aggregator import CLSAggregator


def load_aggregator(checkpoint_path):
    """Load frozen CLSAggregator."""
    aggregator = CLSAggregator(hidden_size=768)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "aggregator" in ckpt:
        state_dict = ckpt["aggregator"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    cleaned = {k.removeprefix("_orig_mod.").removeprefix("aggregator."): v
               for k, v in state_dict.items()}
    aggregator.load_state_dict(cleaned)
    aggregator.eval()
    return aggregator


@torch.no_grad()
def embed_documents(aggregator, doc_cls, doc_rates, cls_budget, device):
    """Run aggregator on cached CLS tokens → (N, D) embeddings + (N,) targets."""
    from final_training_pipeline.train import form_doc_batches

    batches = form_doc_batches(doc_cls, doc_rates, cls_budget, shuffle=False)
    all_embs = []
    all_targets = []

    for cls_tokens, attention_mask, targets in batches:
        cls_tokens = cls_tokens.to(device)
        attention_mask = attention_mask.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            doc_emb = aggregator(cls_tokens, attention_mask=attention_mask)
        all_embs.append(doc_emb.float().cpu())
        all_targets.append(targets)

    X = torch.cat(all_embs, dim=0).numpy()
    y = torch.cat(all_targets, dim=0).numpy()
    return X, y


def ridge_regression(X_train, y_train, alpha=0.0):
    """Closed-form ridge: w = (X^T X + αI)^{-1} X^T y (with bias via augmentation)."""
    # Augment with bias column
    N, D = X_train.shape
    X_aug = np.hstack([X_train, np.ones((N, 1))])

    A = X_aug.T @ X_aug
    if alpha > 0:
        A += alpha * np.eye(D + 1)

    w = np.linalg.solve(A, X_aug.T @ y_train)
    return w


def predict(X, w):
    X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
    return X_aug @ w


def main():
    parser = argparse.ArgumentParser(description="OLS baseline with frozen aggregator")
    parser.add_argument("--cache-dir", required=True,
                        help="Directory containing cls_cache_epoch*_*.pt files")
    parser.add_argument("--aggregator-checkpoint", required=True)
    parser.add_argument("--epoch", type=int, default=0,
                        help="Which epoch's cache to use")
    parser.add_argument("--cls-budget", type=int, default=512)
    parser.add_argument("--alphas", type=str, default="0,0.01,0.1,1,10,100,1000",
                        help="Comma-separated ridge alphas to try")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alphas = [float(a) for a in args.alphas.split(",")]

    # Load caches
    train_cache = os.path.join(args.cache_dir, f"cls_cache_epoch{args.epoch}_train.pt")
    val_cache = os.path.join(args.cache_dir, f"cls_cache_epoch{args.epoch}_val.pt")

    print("Loading CLS caches...")
    t0 = time.time()
    train_cached = torch.load(train_cache, map_location="cpu", weights_only=False)
    val_cached = torch.load(val_cache, map_location="cpu", weights_only=False)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Load aggregator
    print("Loading aggregator...")
    aggregator = load_aggregator(args.aggregator_checkpoint).to(device)

    # Compute document embeddings
    print("Computing train embeddings...")
    t0 = time.time()
    X_train, y_train = embed_documents(
        aggregator, train_cached["doc_cls"], train_cached["doc_rates"],
        args.cls_budget, device)
    print(f"  {X_train.shape[0]} docs, {X_train.shape[1]}D in {time.time() - t0:.1f}s")

    print("Computing val embeddings...")
    t0 = time.time()
    X_val, y_val = embed_documents(
        aggregator, val_cached["doc_cls"], val_cached["doc_rates"],
        args.cls_budget, device)
    print(f"  {X_val.shape[0]} docs, {X_val.shape[1]}D in {time.time() - t0:.1f}s")

    del aggregator, train_cached, val_cached
    torch.cuda.empty_cache()

    # Baselines
    train_mean = y_train.mean()
    val_mse_mean = ((y_val - train_mean) ** 2).mean()
    val_mae_mean = np.abs(y_val - train_mean).mean()
    train_mse_mean = ((y_train - train_mean) ** 2).mean()

    print(f"\n{'='*60}")
    print(f"Predict-mean baseline:  val MSE={val_mse_mean:.4f}, MAE={val_mae_mean:.4f}")
    print(f"{'='*60}")

    # Ridge sweep
    print(f"\n{'alpha':>10} | {'train MSE':>10} {'train MAE':>10} | {'val MSE':>10} {'val MAE':>10} {'val R²':>8}")
    print("-" * 70)

    for alpha in alphas:
        w = ridge_regression(X_train, y_train, alpha=alpha)

        pred_train = predict(X_train, w)
        train_mse = ((pred_train - y_train) ** 2).mean()
        train_mae = np.abs(pred_train - y_train).mean()

        pred_val = predict(X_val, w)
        val_mse = ((pred_val - y_val) ** 2).mean()
        val_mae = np.abs(pred_val - y_val).mean()
        val_r2 = 1 - val_mse / val_mse_mean

        print(f"{alpha:>10.2f} | {train_mse:>10.4f} {train_mae:>10.4f} | "
              f"{val_mse:>10.4f} {val_mae:>10.4f} {val_r2:>8.4f}")

    print(f"\n(For reference: gradient descent got val MSE=0.0072, MAE=0.0621)")


if __name__ == "__main__":
    main()
