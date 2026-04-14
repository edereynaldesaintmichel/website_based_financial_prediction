# Remote Deployment — CLS Aggregator Training (Latent MLM)

Trains a transformer aggregator over per-chunk CLS embeddings using a JEPA-style
latent MLM objective: MSE on masked-position predictions against a frozen
`sg(W·cls)` teacher, plus a SIGReg regularizer that forces the prediction
distribution toward a standard Gaussian (mitigates anisotropy compounding).

## Prerequisites

- `HF_TOKEN` with read+write access to the `edereynal/financial_prediction`
  dataset. Read is needed to pull `documents.pt`, the T5 checkpoint, and any
  previously-computed CLS caches / aggregator checkpoints. Write is needed
  because training auto-uploads each freshly-computed CLS cache shard back to
  HF in a background thread.

## Steps

1. **Connect** via SSH (details provided per-session).

2. **Clone and run setup** on the remote. `setup.sh` unconditionally fetches
   `documents.pt`, the T5 checkpoint, and any existing `aggregator/*` artifacts
   (CLS caches, previously-trained `aggregator.pt`). Missing aggregator
   artifacts are tolerated — first run populates them.
   ```
   export HF_TOKEN=hf_REMOVED
   cd /workspace && git clone https://github.com/edereynaldesaintmichel/website_based_financial_prediction.git
   cd website_based_financial_prediction && bash cls_aggregator_training_pipeline/setup.sh
   ```

   Populates:
   - `/workspace/data/documents.pt`
   - `/workspace/data/t5_checkpoint/full_model.pt`
   - `/workspace/data/aggregator/cls_cache/*.pt` (if present on HF)
   - `/workspace/data/aggregator/aggregator.pt` (if present on HF)

3. **Check GPUs**:
   ```
   nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
   ```

4. **Token budgets** (per-GPU; DDP replicates the model):
   - T5 encoder (~300M) ≈ 2 GB bf16; aggregator (~113M) ≈ 1 GB.
   - `--encoder_token_budget`: total tokens per encoder batch while computing
     CLS embeddings (only used when cache missing).
   - `--cls_token_budget`: padded CLS tokens (B × max_n_cls) per aggregator
     batch.
   - Starting points:
     - **192 GB B200**: `--encoder_token_budget 65536 --cls_token_budget 65536`
     - **32 GB 5090**: `--encoder_token_budget 16384 --cls_token_budget 16384`
   - If OOM, halve `cls_token_budget` first and bump `--grad_accum_steps`.

5. **Training command** (user runs in their SSH session — do NOT run yourself):

   **Single GPU:**
   ```
   cd /workspace/website_based_financial_prediction && python \
       -m cls_aggregator_training_pipeline.train \
       --data /workspace/data/documents.pt \
       --checkpoint /workspace/data/t5_checkpoint/full_model.pt \
       --output_dir /workspace/data/aggregator \
       --cls_cache_dir /workspace/data/aggregator/cls_cache \
       --encoder_token_budget 65536 \
       --cls_token_budget 32768 \
       --grad_accum_steps 1 \
       --lr 3e-4 \
       --muon_lr 0.02 \
       --mask_prob 0.2 \
       --sigreg_lambda 1.0 \
       --epochs 5
   ```

   **Multi-GPU (e.g. 4×5090):**
   ```
   cd /workspace/website_based_financial_prediction && torchrun \
       --nproc_per_node=4 \
       -m cls_aggregator_training_pipeline.train \
       --data /workspace/data/documents.pt \
       --checkpoint /workspace/data/t5_checkpoint/full_model.pt \
       --output_dir /workspace/data/aggregator \
       --cls_cache_dir /workspace/data/aggregator/cls_cache \
       --encoder_token_budget 16384 \
       --cls_token_budget 8192 \
       --grad_accum_steps 1 \
       --lr 3e-4 \
       --muon_lr 0.02 \
       --mask_prob 0.2 \
       --sigreg_lambda 1.0 \
       --epochs 5
   ```
   Pointing `--output_dir` / `--cls_cache_dir` at `/workspace/data/aggregator*`
   keeps everything under the tree `setup.sh` already mirrors from HF.

6. **CLS cache auto-upload.** When a cache shard is missing, rank 0 computes
   it, saves to `--cls_cache_dir`, and spawns a background daemon thread that
   uploads the file to `aggregator/cls_cache/<basename>` on HF. Upload failures
   are logged and non-fatal. On subsequent runs `setup.sh` will pull caches
   back down, skipping re-encoding entirely.

7. **Retrieve the aggregator checkpoint** after training:
   ```
   hf upload edereynal/financial_prediction \
       /workspace/data/aggregator/latest.pt \
       aggregator/aggregator.pt --repo-type dataset
   ```

## Resuming

Re-run the same command with `--resume`. The checkpoint at
`{output_dir}/latest.pt` restores aggregator + optimizer (Muon + AdamW) +
scheduler state.

## Notes

- Optimizer is `CombinedOptimizer([Muon(2D params), AdamW(1D params)])`. Muon
  handles `Wqkv`, `Wo`, `w_gate/up/down`, and the shared projection `W`;
  AdamW handles LayerNorm gains/biases and the learnable `mask_token`.
- `--lr` controls AdamW; `--muon_lr` controls Muon (default 0.02). A single
  cosine schedule with `--warmup_steps` warmup wraps both.
- Aggregator output at inference: `(B, N, D)` per-position embeddings — caller
  mean-pools over valid positions to get the document embedding.
- The T5 encoder is loaded lazily: if both train and val CLS caches exist for
  the current epoch, the encoder is never instantiated.
- SSH + `HF_TOKEN` per-session, as before; PyTorch + CUDA pre-installed on
  vast.ai instances.
