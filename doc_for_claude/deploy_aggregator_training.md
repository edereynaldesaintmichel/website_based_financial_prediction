# Remote Deployment Instructions — CLS Aggregator Training

## Prerequisites

- `HF_TOKEN` with read access to the `edereynal/financial_prediction` dataset. Both `documents.pt` and the T5 checkpoint are pulled from there.

## Steps

1. **Connect** via SSH (connection details provided by user each session).

2. **Clone repo and run setup** on the remote. The setup script downloads `documents.pt` and the T5 checkpoint into `/workspace/data/`:
   ```
   export HF_TOKEN=<user-provided>
   cd /workspace && git clone https://github.com/edereynaldesaintmichel/website_based_financial_prediction.git
   cd website_based_financial_prediction && bash cls_aggregator_training_pipeline/setup.sh
   ```

   Result:
   - `/workspace/data/documents.pt`
   - `/workspace/data/t5_checkpoint/full_model.pt`

3. **Check GPUs** on the remote:
   ```
   nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
   ```
   Examples: 1× B200 (192 GB), 4×5090 (32 GB each), 8×5090.

4. **Compute token budgets** based on available VRAM **per GPU**:
   - The T5 model (~300M params) uses ~2 GB in bf16.
   - The aggregator (~56M params) adds ~0.5 GB.
   - Each GPU holds a full model replica (DDP); budget must fit in per-GPU VRAM.
   - Recommended starting points:
     - **192 GB B200**: `--encoder_token_budget 65536`, `--decoder_token_budget 32768`
     - **32 GB 5090**: `--encoder_token_budget 16384`, `--decoder_token_budget 4096`
   - If OOM, halve `decoder_token_budget` first and double `grad_accum_steps` to compensate.

5. **Give the user the training command** to run in their remote terminal.

   **Single GPU:**
   ```
   cd /workspace/website_based_financial_prediction && python \
       -m cls_aggregator_training_pipeline.train \
       --data /workspace/data/documents.pt \
       --checkpoint /workspace/data/t5_checkpoint/full_model.pt \
       --output_dir /workspace/checkpoints/cls_aggregator \
       --encoder_token_budget 65536 \
       --decoder_token_budget 32768 \
       --grad_accum_steps 1 \
       --lr 7e-5 \
       --epochs 3
   ```

   **Multi-GPU (e.g. 4×5090):**
   ```
   cd /workspace/website_based_financial_prediction && torchrun \
       --nproc_per_node=4 \
       -m cls_aggregator_training_pipeline.train \
       --data /workspace/data/documents.pt \
       --checkpoint /workspace/data/t5_checkpoint/full_model.pt \
       --output_dir /workspace/checkpoints/cls_aggregator \
       --encoder_token_budget 16384 \
       --decoder_token_budget 4096 \
       --grad_accum_steps 1 \
       --lr 7e-5 \
       --epochs 3
   ```
   Adjust `--nproc_per_node` to match the number of GPUs.

   Do NOT run this command yourself. The user runs it directly in their SSH session.

6. **Retrieve checkpoints** after training — either `rsync` back locally, or push to the HF dataset from the remote:
   ```
   hf upload edereynal/financial_prediction /workspace/checkpoints/cls_aggregator/aggregator.pt aggregator/aggregator.pt --repo-type dataset
   ```

## Resuming

If training is interrupted, re-run the same command with `--resume`.

## Notes

- SSH connection details change daily — always provided by the user at the start of each session.
- `HF_TOKEN` must be exported before running `setup.sh`.
- The remote instance is typically a vast.ai GPU instance with PyTorch + CUDA pre-installed.
- The `documents.pt` file is produced once by the MLM pipeline and shared across all training pipelines.
