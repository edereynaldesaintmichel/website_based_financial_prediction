# Remote Deployment Instructions -- T5-Style CLS Training

## Prerequisites

- A trained MLM encoder checkpoint must exist locally at `checkpoints/mlm_full_baseline/<checkpoint_dir>/full_model.pt`.
- `documents.pt` must exist (shared with MLM and CLS aggregator pipelines). If not already on the remote, it needs to be uploaded.

## Steps

1. **Connect** via SSH (connection details provided by user each session).

2. **Clone repo and run setup** on the remote:
   ```
   cd /workspace && git clone https://github.com/edereynaldesaintmichel/website_based_financial_prediction.git
   cd website_based_financial_prediction && bash t5_style_training_pipeline/setup.sh
   ```

3. **Check GPU VRAM** on the remote:
   ```
   nvidia-smi --query-gpu=memory.total --format=csv,noheader
   ```

4. **Create data directories** on the remote:
   ```
   ssh -p <PORT> root@<HOST> "mkdir -p /workspace/data/encoder_checkpoint"
   ```

5. **Upload `documents.pt`** (only if not already present on the remote):
   ```
   ssh -p <PORT> root@<HOST> "ls /workspace/data/documents.pt" 2>/dev/null || \
     rsync -avz --progress -e "ssh -p <PORT>" mlm_data/documents.pt root@<HOST>:/workspace/data/documents.pt
   ```

6. **Upload encoder checkpoint** to remote `/workspace/data/encoder_checkpoint/`:
   ```
   rsync -avz --progress -e "ssh -p <PORT>" checkpoints/mlm_full_baseline/<CHECKPOINT_DIR>/full_model.pt root@<HOST>:/workspace/data/encoder_checkpoint/
   ```

7. **Compute batch size and learning rate** based on available VRAM:
   - Reference point: **8192 tokens_per_batch** uses ~54 GB VRAM on H100, with **lr=5e-5**.
   - VRAM scales linearly with batch size. LR scales linearly with batch size too.
   - Example (assuming ~8 GB overhead for model weights):
     - 48 GB VRAM -> tokens_per_batch=10240, lr=6.25e-5

8. **Give the user the training command** to run in their remote terminal:
   ```
   cd /workspace/website_based_financial_prediction && python -m t5_style_training_pipeline.train --data /workspace/data/documents.pt --encoder_checkpoint /workspace/data/encoder_checkpoint/full_model.pt --tokens_per_batch <BATCH> --lr <LR> --compile
   ```
   Do NOT run this command yourself. The user runs it directly in their SSH session.

## Notes

- SSH connection details change daily -- always provided by the user at the start of each session.
- `documents.pt` is shared across MLM, T5, and CLS aggregator pipelines -- only upload once.
- The remote instance is typically a vast.ai GPU instance with PyTorch + CUDA pre-installed.
- The encoder checkpoint upload is a one-time step per encoder version.
- Contrastive loss lambda (default 0.02) and encoder dropout (default 0.1) can be tuned via `--contrastive_lambda` and `--encoder_dropout` flags. Both losses (reconstruction and contrastive) are logged separately.
