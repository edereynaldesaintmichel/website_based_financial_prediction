# Remote Deployment Instructions — CLS Aggregator Training (single B200)

## Prerequisites

- A trained T5 checkpoint must exist locally at `checkpoints/t5_expanded_memory/model_only.pt`.
- Raw markdown documents must exist locally in:
  - `training_data/processed/SEC_10k_markdown_tagged/`
  - `training_data/processed/companies_house_markdown_tagged/`

## Steps

1. **Connect** via SSH (connection details provided by user each session).

2. **Clone repo and run setup** on the remote:
   ```
   cd /workspace && git clone https://github.com/edereynaldesaintmichel/website_based_financial_prediction.git
   cd website_based_financial_prediction && bash t5_style_training_pipeline/setup.sh
   ```

3. **Check GPUs** on the remote:
   ```
   nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
   ```
   Expect 1× B200 with 192 GB.

4. **Upload raw markdown documents** to the remote:
   ```
   rsync -avz --progress -e "ssh -p <PORT>" \
       training_data/processed/SEC_10k_markdown_tagged/ \
       root@<HOST>:/workspace/data/SEC_10k_markdown_tagged/

   rsync -avz --progress -e "ssh -p <PORT>" \
       training_data/processed/companies_house_markdown_tagged/ \
       root@<HOST>:/workspace/data/companies_house_markdown_tagged/
   ```

5. **Upload T5 checkpoint** to the remote:
   ```
   rsync -avz --progress -e "ssh -p <PORT>" \
       checkpoints/t5_expanded_memory/model_only.pt \
       root@<HOST>:/workspace/data/t5_checkpoint/model_only.pt
   ```

6. **Tokenize documents on the remote** (runs in parallel, ~minutes):
   ```
   cd /workspace/website_based_financial_prediction && python -m cls_aggregator_training_pipeline.prepare_dataset \
       --input_dirs /workspace/data/SEC_10k_markdown_tagged \
                    /workspace/data/companies_house_markdown_tagged \
       --output /workspace/data/cls_aggregator/documents.pt
   ```
   Wait for this to complete before starting training.

7. **Compute token budgets** based on available VRAM:
   - The T5 model (~300M params) uses ~2 GB in bf16.
   - The aggregator (~56M params) adds ~0.5 GB.
   - That leaves ~189 GB for activations and batch data.
   - Recommended starting point for 192 GB B200:
     - `--encoder_token_budget 65536` (for CLS computation, no gradients)
     - `--decoder_token_budget 32768` (for training, holds gradients)
   - If OOM, halve `decoder_token_budget` first and double `grad_accum_steps` to compensate.

8. **Give the user the training command** to run in their remote terminal:
   ```
   cd /workspace/website_based_financial_prediction && python \
       -m cls_aggregator_training_pipeline.train \
       --data /workspace/data/cls_aggregator/documents.pt \
       --checkpoint /workspace/data/t5_checkpoint/model_only.pt \
       --output_dir /workspace/checkpoints/cls_aggregator \
       --encoder_token_budget 65536 \
       --decoder_token_budget 32768 \
       --grad_accum_steps 1 \
       --lr 7e-5 \
       --epochs 3
   ```
   Do NOT run this command yourself. The user runs it directly in their SSH session.

9. **Download checkpoints** after training (from local machine):
   ```
   rsync -avz --progress -e "ssh -p <PORT>" \
       root@<HOST>:/workspace/checkpoints/cls_aggregator/ \
       checkpoints/cls_aggregator/
   ```

## Resuming

If training is interrupted, re-run the same command with `--resume`:
```
cd /workspace/website_based_financial_prediction && python \
   -m cls_aggregator_training_pipeline.train \
   --data /workspace/data/cls_aggregator/documents.pt \
   --checkpoint /workspace/data/t5_checkpoint/model_only.pt \
   --output_dir /workspace/checkpoints/cls_aggregator \
   --resume \
   --encoder_token_budget 65536 \
   --decoder_token_budget 32768 \
   --grad_accum_steps 1 \
   --lr 7e-5 \
   --epochs 3
```

## Notes

- SSH connection details change daily — always provided by the user at the start of each session.
- The remote instance is typically a vast.ai GPU instance with PyTorch + CUDA pre-installed.
- The tokenization step (step 6) only needs to run once; the `.pt` file persists.
