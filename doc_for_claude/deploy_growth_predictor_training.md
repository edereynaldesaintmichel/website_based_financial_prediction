# Remote Deployment — Growth Predictor Training

Trains a small regression head (and optionally fine-tunes the aggregator) on
top of the pre-computed chunk-level CLS embeddings produced by the aggregator
pipeline. The JEPA-trained `CLSAggregator` emits per-position outputs
`(B, N, D)`; `GrowthPredictor` mean-pools over valid positions to get a
document embedding, then regresses a 5-year growth rate (MSE loss). No
encoder runs at train time — all inputs come from cached CLS tensors on HF.

## Prerequisites

- `HF_TOKEN` with read access to the `edereynal/financial_prediction`
  dataset. Read is needed to pull `documents.pt`, `growth_rates.json`, the
  JEPA-trained aggregator checkpoint, and the CLS cache shards produced by
  the aggregator pipeline. Write is only needed if you plan to upload the
  trained predictor back to HF.

## Steps

1. **Connect** via SSH (details provided per-session).

2. **Clone, rsync setup script, and run setup** on the remote. `setup.sh` is
   gitignored (contains the HF token), so rsync it from the local repo to
   the remote after cloning. `setup.sh` pulls every artifact needed to train
   from cached CLS embeddings — no T5 encoder is downloaded.

   On the remote:
   ```
   cd /workspace && git clone https://github.com/edereynaldesaintmichel/website_based_financial_prediction.git
   ```

   From your local machine (project root):
   ```
   rsync -avz final_training_pipeline/setup.sh \
       <user>@<remote>:/workspace/website_based_financial_prediction/final_training_pipeline/setup.sh
   ```

   Back on the remote:
   ```
   cd /workspace/website_based_financial_prediction && bash final_training_pipeline/setup.sh
   ```

   Populates:
   - `/workspace/data/documents.pt`
   - `/workspace/data/growth_rates.json`
   - `/workspace/data/aggregator/aggregator_jepa.pt` (JEPA-trained weights)
   - `/workspace/data/aggregator/cls_cache/cls_cache_{train,val}_epoch{0..4}.pt`
   - Symlinks under `/workspace/data/growth_predictor/` pointing at each
     cache shard, so `ensure_cache()` finds them locally without touching HF.

3. **Check GPUs**:
   ```
   nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
   ```

4. **Token budgets**:
   - Aggregator (~113M) + head (<1M) ≈ 1 GB bf16. Memory is dominated by the
     padded CLS batch, not model weights.
   - `--cls-budget`: max total (padded) CLS embeddings per predictor batch.
     Higher values = better pooling signal but more memory.
   - Starting points:
     - **192 GB B200**: `--cls-budget 2048`
     - **32 GB 5090**:  `--cls-budget 1024`
   - If OOM during the forward pass, halve `--cls-budget` and bump
     `--grad-accum-steps`.

5. **Training command** (user runs in their SSH session — do NOT run yourself):

   **Frozen aggregator (train head only):**
   ```
   cd /workspace/website_based_financial_prediction && python \
       -m final_training_pipeline.train \
       --data /workspace/data/documents.pt \
       --growth-rates /workspace/data/growth_rates.json \
       --aggregator-checkpoint /workspace/data/aggregator/aggregator_jepa.pt \
       --output-dir /workspace/data/growth_predictor \
       --cls-budget 2048 \
       --lr 1e-4 \
       --epochs 10 \
       --compile
   ```

   **Trainable aggregator (fine-tune aggregator + head):**
   ```
   cd /workspace/website_based_financial_prediction && python \
       -m final_training_pipeline.train \
       --data /workspace/data/documents.pt \
       --growth-rates /workspace/data/growth_rates.json \
       --aggregator-checkpoint /workspace/data/aggregator/aggregator_jepa.pt \
       --output-dir /workspace/data/growth_predictor \
       --cls-budget 2048 \
       --lr 1e-4 \
       --train-aggregator \
       --aggregator-lr 1e-6 \
       --regularization \
       --reg-lambda 0.1 \
       --epochs 10 \
       --compile
   ```

   Pointing `--output-dir` at `/workspace/data/growth_predictor` means the
   cache symlinks created by `setup.sh` are found immediately, and no HF
   download happens at epoch boundaries.

6. **CLS cache alignment.** Caches are `dict[doc_idx → Tensor(n_chunks, D)]`
   where `doc_idx` is the position in `split_documents(documents.pt,
   val_ratio)`. The training script reloads `documents.pt`, reruns the same
   split, and builds `{doc_idx → growth_rate}` by extracting `company_id`
   from each document's `source_file`. **`--val_ratio` must match the value
   used by the aggregator pipeline when the caches were generated**
   (default `0.1`), otherwise `doc_idx` keys won't align. Caches cycle every
   5 epochs (`NUM_CACHED_EPOCHS = 5`); a missing shard is fetched from
   `aggregator/cls_cache/<filename>` on HF automatically.

7. **Retrieve the predictor checkpoint** after training — either `rsync`
   back locally, or push to HF:
   ```
   hf upload edereynal/financial_prediction \
       /workspace/data/growth_predictor/checkpoint_latest/full_model.pt \
       growth_predictor/full_model.pt --repo-type dataset
   ```

## Resuming

If training is interrupted, re-run the same command with
`--resume-from /workspace/data/growth_predictor/checkpoint_latest`. The
checkpoint restores predictor weights + optimizer + scheduler state.

## Notes

- Single-GPU training script (no DDP). The aggregator is small enough that
  sharding isn't useful here; the bottleneck is loading/aligning caches.
- Optimizer is `AdamW` with separate param groups for head (`--lr`) and, if
  `--train-aggregator` is set, aggregator (`--aggregator-lr`, default `1e-6`).
  Cosine schedule with `--warmup-ratio` warmup.
- `--regularization` adds MSE between the current aggregator's mean-pooled
  document embedding and a reference embedding computed once at startup from
  the initial aggregator weights — keeps the aggregator close to its
  JEPA-trained representation while the head trains.
- Training stops when the cosine LR schedule reaches zero, even if not all
  epochs are completed.
- To refresh `growth_rates.json`, run `python -m final_training_pipeline.prepare_data`
  locally, then `hf upload edereynal/financial_prediction growth_rates.json growth_rates.json --repo-type dataset`.
- SSH + `HF_TOKEN` per-session, as before; PyTorch + CUDA pre-installed on
  vast.ai instances.
