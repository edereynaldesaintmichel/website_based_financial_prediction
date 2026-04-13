# Remote Deployment Instructions — T5-Style CLS Training

## Prerequisites

- `HF_TOKEN` with read access to the `edereynal/financial_prediction` dataset. All artifacts (`documents.pt`, MLM encoder checkpoint) are pulled from there.

## Steps

1. **Connect** via SSH (connection details provided by user each session).

2. **Clone repo and run setup** on the remote. The setup script downloads `documents.pt` and the MLM encoder checkpoint into `/workspace/data/`:
   ```
   export HF_TOKEN=<user-provided>
   cd /workspace && git clone https://github.com/edereynaldesaintmichel/website_based_financial_prediction.git
   cd website_based_financial_prediction && bash t5_style_training_pipeline/setup.sh
   ```

   Result:
   - `/workspace/data/documents.pt`
   - `/workspace/data/encoder_checkpoint/full_model.pt`

3. **Check GPU VRAM** on the remote:
   ```
   nvidia-smi --query-gpu=memory.total --format=csv,noheader
   ```

4. **Compute batch size and learning rate** based on available VRAM:
   - Reference point: **16384 tokens_per_batch** uses ~98 GB VRAM on RTX 6000 Blackwell, with **lr=1e-4**.
   - VRAM scales linearly with batch size. LR scales linearly with batch size too.
   - Example (assuming ~8 GB overhead for model weights):
     - 48 GB VRAM -> tokens_per_batch=6553, lr=4e-5

5. **Give the user the training command** to run in their remote terminal:
   ```
   cd /workspace/website_based_financial_prediction && nohup python -m t5_style_training_pipeline.train --data /workspace/data/documents.pt --encoder_checkpoint /workspace/data/encoder_checkpoint/full_model.pt --tokens_per_batch <BATCH> --lr <LR> --compile > train.log 2>&1 &
   ```
   Do NOT run this command yourself. The user runs it directly in their SSH session.

## Notes

- SSH connection details change daily — always provided by the user at the start of each session.
- `HF_TOKEN` must be exported before running `setup.sh`.
- The remote instance is typically a vast.ai GPU instance with PyTorch + CUDA pre-installed.
- Contrastive loss lambda (default 0.02) and encoder dropout (default 0.1) can be tuned via `--contrastive_lambda` and `--encoder_dropout` flags. Both losses (reconstruction and contrastive) are logged separately.
