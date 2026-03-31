# Remote Deployment Instructions — Growth Predictor Training

## Prerequisites

- A trained T5 checkpoint must exist locally at `checkpoints/t5_cls/checkpoint_epoch5/full_model.pt`.
- A trained CLS aggregator checkpoint must exist locally at `checkpoints/cls_aggregator/aggregator.pt` (aggregator-only weights).
- A pre-tokenized `documents.pt` file must exist (produced by the MLM training pipeline's `prepare_data.py`).
- A merged `growth_rates.json` file must exist (produced by `python -m final_training_pipeline.prepare_data`).

## Steps

1. **Connect** via SSH (connection details provided by user each session).

2. **Clone repo and run setup** on the remote:
   ```
   cd /workspace && git clone https://github.com/edereynaldesaintmichel/website_based_financial_prediction.git
   ```

3. **Check GPUs** on the remote:
   ```
   nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
   ```

4. **Upload data files** to the remote:
   ```
   ssh -p <PORT> root@<HOST> "mkdir -p /workspace/data/t5_checkpoint /workspace/data/aggregator_checkpoint"

   rsync -avz --progress -e "ssh -p <PORT>" \
       mlm_data/documents.pt \
       root@<HOST>:/workspace/data/documents.pt

   rsync -avz --progress -e "ssh -p <PORT>" \
       growth_rates.json \
       root@<HOST>:/workspace/data/growth_rates.json
   ```

5. **Upload model checkpoints** to the remote:
   ```
   rsync -avz --progress -e "ssh -p <PORT>" \
       checkpoints/t5_cls/checkpoint_epoch5/full_model.pt \
       root@<HOST>:/workspace/data/t5_checkpoint/full_model.pt

   rsync -avz --progress -e "ssh -p <PORT>" \
       checkpoints/cls_aggregator/aggregator.pt \
       root@<HOST>:/workspace/data/aggregator_checkpoint/aggregator.pt
   ```

6. **Compute token budgets** based on available VRAM:
   - The T5 encoder (~150M params, frozen) uses ~1 GB in bf16.
   - The GrowthPredictor (aggregator + head) is small (<60M params).
   - `--token-budget` controls encoder CLS computation batching (same as aggregator training).
   - `--cls-budget` controls the document batching for the predictor head — max total CLS embeddings per batch.
   - Recommended starting points:
     - **192 GB B200**: `--token-budget 65536`, `--cls-budget 1024`
     - **32 GB 5090**: `--token-budget 16384`, `--cls-budget 512`
   - If OOM during CLS computation, halve `--token-budget`.
   - If OOM during predictor forward pass, halve `--cls-budget`.

7. **Give the user the training command** to run in their remote terminal.

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

8. **Download checkpoints** after training (from local machine):
   ```
   rsync -avz --progress -e "ssh -p <PORT>" \
       root@<HOST>:/workspace/checkpoints/growth_predictor/ \
       checkpoints/growth_predictor/
   ```

## Resuming

If training is interrupted, re-run the same command with `--resume-from`:
```
cd /workspace/website_based_financial_prediction && python \
    -m final_training_pipeline.train \
    --data /workspace/data/documents.pt \
    --growth-rates /workspace/data/growth_rates.json \
    --encoder-checkpoint /workspace/data/t5_checkpoint/full_model.pt \
    --aggregator-checkpoint /workspace/data/aggregator_checkpoint/aggregator.pt \
    --output-dir /workspace/checkpoints/growth_predictor \
    --resume-from /workspace/checkpoints/growth_predictor/checkpoint_latest \
    --token-budget 65536 \
    --cls-budget 1024 \
    --lr 1e-4 \
    --epochs 10 \
    --compile
```

## Notes

- This is a single-GPU training script (no DDP) — the encoder is frozen and only the predictor head (and optionally the aggregator) are trained.
- SSH connection details change daily — always provided by the user at the start of each session.
- The remote instance is typically a vast.ai GPU instance with PyTorch + CUDA pre-installed.
- Training stops automatically when the cosine LR schedule reaches zero, even if not all epochs are completed.
- The `growth_rates.json` file is produced locally by `python -m final_training_pipeline.prepare_data` (merges UK and SEC rates).
