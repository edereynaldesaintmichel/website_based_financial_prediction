# Remote Deployment Instructions — Growth Predictor Training

## Prerequisites

- `HF_TOKEN` with read access to the `edereynal/financial_prediction` dataset. All artifacts (`documents.pt`, `growth_rates.json`, T5 checkpoint, aggregator checkpoint) are pulled from there.

## Steps

1. **Connect** via SSH (connection details provided by user each session).

2. **Clone repo and run setup** on the remote. The setup script downloads every artifact needed into `/workspace/data/`:
   ```
   export HF_TOKEN=<user-provided>
   cd /workspace && git clone https://github.com/edereynaldesaintmichel/website_based_financial_prediction.git
   cd website_based_financial_prediction && bash final_training_pipeline/setup.sh
   ```

   Result:
   - `/workspace/data/documents.pt`
   - `/workspace/data/growth_rates.json`
   - `/workspace/data/t5_checkpoint/full_model.pt`
   - `/workspace/data/aggregator_checkpoint/aggregator.pt`

3. **Check GPUs** on the remote:
   ```
   nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
   ```

4. **Compute token budgets** based on available VRAM:
   - The T5 encoder (~150M params, frozen) uses ~1 GB in bf16.
   - The GrowthPredictor (aggregator + head) is small (<60M params).
   - `--token-budget` controls encoder CLS computation batching (same as aggregator training).
   - `--cls-budget` controls the document batching for the predictor head — max total CLS embeddings per batch.
   - Recommended starting points:
     - **192 GB B200**: `--token-budget 65536`, `--cls-budget 1024`
     - **32 GB 5090**: `--token-budget 16384`, `--cls-budget 512`
   - If OOM during CLS computation, halve `--token-budget`.
   - If OOM during predictor forward pass, halve `--cls-budget`.

5. **Give the user the training command** to run in their remote terminal.

   **Frozen aggregator (train head only):**
   ```
   cd /workspace/website_based_financial_prediction && python \
       -m final_training_pipeline.train \
       --data /workspace/data/documents.pt \
       --growth-rates /workspace/data/growth_rates.json \
       --encoder-checkpoint /workspace/data/t5_checkpoint/full_model.pt \
       --aggregator-checkpoint /workspace/data/aggregator_checkpoint/aggregator.pt \
       --output-dir /workspace/checkpoints/growth_predictor \
       --token-budget 65536 \
       --cls-budget 1024 \
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
       --encoder-checkpoint /workspace/data/t5_checkpoint/full_model.pt \
       --aggregator-checkpoint /workspace/data/aggregator_checkpoint/aggregator.pt \
       --output-dir /workspace/checkpoints/growth_predictor \
       --token-budget 65536 \
       --cls-budget 1024 \
       --lr 1e-4 \
       --train-aggregator \
       --aggregator-lr 1e-6 \
       --regularization \
       --reg-lambda 0.1 \
       --epochs 10 \
       --compile
   ```

   Do NOT run this command yourself. The user runs it directly in their SSH session.

6. **Retrieve checkpoints** after training — either `rsync` back locally, or push to the HF dataset from the remote.

## Resuming

If training is interrupted, re-run the same command with `--resume-from /workspace/checkpoints/growth_predictor/checkpoint_latest`.

## Notes

- This is a single-GPU training script (no DDP) — the encoder is frozen and only the predictor head (and optionally the aggregator) are trained.
- SSH connection details change daily — always provided by the user at the start of each session.
- `HF_TOKEN` must be exported before running `setup.sh`.
- The remote instance is typically a vast.ai GPU instance with PyTorch + CUDA pre-installed.
- Training stops automatically when the cosine LR schedule reaches zero, even if not all epochs are completed.
- To refresh `growth_rates.json`, run `python -m final_training_pipeline.prepare_data` locally, then `hf upload edereynal/financial_prediction growth_rates.json growth_rates.json --repo-type dataset`.
