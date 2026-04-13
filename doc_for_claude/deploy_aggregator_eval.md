# Remote Deployment Instructions — CLS Aggregator Embedding Evaluation

## Prerequisites

- `HF_TOKEN` with read access to the `edereynal/financial_prediction` dataset. All artifacts are pulled from there.

## Steps

1. **Connect** via SSH (connection details provided by user each session).

2. **Clone repo and run setup** on the remote in `eval` mode (this also fetches the trained aggregator checkpoint):
   ```
   export HF_TOKEN=<user-provided>
   cd /workspace && git clone https://github.com/edereynaldesaintmichel/website_based_financial_prediction.git
   cd website_based_financial_prediction && bash cls_aggregator_training_pipeline/setup.sh eval
   ```

   Result:
   - `/workspace/data/documents.pt`
   - `/workspace/data/t5_checkpoint/full_model.pt`
   - `/workspace/data/aggregator/aggregator.pt`

3. **Check GPUs** on the remote:
   ```
   nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
   ```

4. **Give the user the eval command** to run in their remote terminal.

   **Single GPU:**
   ```
   cd /workspace/website_based_financial_prediction && python \
       -m cls_aggregator_training_pipeline.eval_aggregator_embeddings \
       --data /workspace/data/documents.pt \
       --t5_checkpoint /workspace/data/t5_checkpoint/full_model.pt \
       --aggregator_checkpoint /workspace/data/aggregator/aggregator.pt \
       --encoder_token_budget 65536 \
       --agg_batch_budget 4096
   ```

   **Multi-GPU (e.g. 4x5090):**
   ```
   cd /workspace/website_based_financial_prediction && torchrun \
       --nproc_per_node=4 \
       -m cls_aggregator_training_pipeline.eval_aggregator_embeddings \
       --data /workspace/data/documents.pt \
       --t5_checkpoint /workspace/data/t5_checkpoint/full_model.pt \
       --aggregator_checkpoint /workspace/data/aggregator/aggregator.pt \
       --encoder_token_budget 16384 \
       --agg_batch_budget 4096
   ```
   Adjust `--nproc_per_node` to match the number of GPUs.

   Do NOT run this command yourself. The user runs it directly in their SSH session.

## Token Budget Guidelines

- The T5 encoder (~150M params) uses ~1 GB in bf16. The aggregator (~56M params) adds ~0.5 GB.
- Each GPU holds a full model replica (DDP); budget must fit in per-GPU VRAM.
- Recommended starting points:
  - **192 GB B200**: `--encoder_token_budget 65536`
  - **32 GB 5090**: `--encoder_token_budget 16384`
- If OOM, halve `encoder_token_budget`.

## Notes

- SSH connection details change daily — always provided by the user at the start of each session.
- `HF_TOKEN` must be exported before running `setup.sh`.
- The remote instance is typically a vast.ai GPU instance with PyTorch + CUDA pre-installed.
- This is an eval-only script — no training happens, no checkpoints are produced. All inference is `@torch.no_grad`.
