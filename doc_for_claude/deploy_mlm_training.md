# Remote Deployment Instructions — MLM Training

## Prerequisites

- `HF_TOKEN` with read access to the `edereynal/financial_prediction` dataset. `documents.pt` is pulled from there.

## Steps

1. **Connect** via SSH (connection details provided by user each session).

2. **Clone repo and run setup** on the remote. The setup script downloads `documents.pt` into `/workspace/data/`:
   ```
   export HF_TOKEN=<user-provided>
   cd /workspace && git clone https://github.com/edereynaldesaintmichel/website_based_financial_prediction.git
   cd website_based_financial_prediction && bash mlm_training_pipeline/setup.sh
   ```

3. **Check GPU VRAM** on the remote:
   ```
   nvidia-smi --query-gpu=memory.total --format=csv,noheader
   ```

4. **Compute batch size and learning rates** based on available VRAM:
   - Reference point: **16384 tokens_per_batch** uses ~54 GB VRAM on H100, with **lr=2.5e-5** and **number_lr=1e-4**.
   - VRAM scales linearly with batch size. LR scales linearly with batch size too.
   - Examples (assuming ~10 GB overhead for model weights):
     - 80 GB VRAM → tokens_per_batch=20480, lr=3.1e-5, number_lr=1.25e-4
     - 48 GB VRAM → tokens_per_batch=11264, lr=1.7e-5, number_lr=6.9e-5
     - 24 GB VRAM → tokens_per_batch=4096, lr=6.25e-6, number_lr=2.5e-5

5. **Give the user the training command** to run in their remote terminal:
   ```
   cd /workspace/website_based_financial_prediction && python -m mlm_training_pipeline.train_mlm_full --data /workspace/data/documents.pt --tokens_per_batch <BATCH> --lr <LR> --number_lr <NUMBER_LR> --dtype bf16 --device cuda --compile
   ```
   Do NOT run this command yourself. The user runs it directly in their SSH session.

## Notes

- SSH connection details change daily — always provided by the user at the start of each session.
- `HF_TOKEN` must be exported before running `setup.sh` (read scope is sufficient on the remote).
- The remote instance is typically a vast.ai GPU instance with PyTorch + CUDA pre-installed.
- The `documents.pt` file includes both financial documents and Wikipedia regularization documents.
