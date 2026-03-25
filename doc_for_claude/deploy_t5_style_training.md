# Remote Deployment Instructions — T5-Style CLS Training

## Prerequisites

- A trained MLM encoder checkpoint must exist locally at `checkpoints/mlm_full_baseline/<checkpoint_dir>/full_model.pt`.
- T5 training data must be prepared locally via `python -m t5_style_training_pipeline.prepare_data`.

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

4. **Upload local T5 training data** from `t5_training_data/` to remote `/workspace/data/t5/`:
   - Wipe the remote directory first (`rm -rf /workspace/data/t5/*`)
   - Then rsync all `.pt` files up:
     ```
     rsync -avz --progress -e "ssh -p <PORT>" t5_training_data/ root@<HOST>:/workspace/data/t5/
     ```

5. **Upload encoder checkpoint** to remote `/workspace/data/encoder_checkpoint/`:
   ```
   rsync -avz --progress -e "ssh -p <PORT>" checkpoints/mlm_full_baseline/<CHECKPOINT_DIR>/full_model.pt root@<HOST>:/workspace/data/encoder_checkpoint/
   ```

6. **Compute batch size and learning rate** based on available VRAM:
   - Reference point: **8192 tokens_per_batch** uses ~54 GB VRAM on H100, with **lr=5e-5**.
   - VRAM scales linearly with batch size. LR scales linearly with batch size too.
   - Example (assuming ~8 GB overhead for model weights):
     - 48 GB VRAM → tokens_per_batch=10240, lr=6.25e-5

7. **Give the user the training command** to run in their remote terminal:
   ```
   cd /workspace/website_based_financial_prediction && python -m t5_style_training_pipeline.train --data_dir /workspace/data/t5 --encoder_checkpoint /workspace/data/encoder_checkpoint/full_model.pt --tokens_per_batch <BATCH> --lr <LR> --num_workers 4 --compile
   ```
   Do NOT run this command yourself. The user runs it directly in their SSH session.

## Notes

- SSH connection details change daily — always provided by the user at the start of each session.
- Wait for all local data files to be present before uploading.
- The remote instance is typically a vast.ai GPU instance with PyTorch + CUDA pre-installed.
- The encoder checkpoint upload is a one-time step per encoder version.
