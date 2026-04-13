#!/usr/bin/env bash
# Setup script for T5-style training on vast.ai.
# Assumes pytorch + cuda are already installed.
set -e

export HF_TOKEN=hf_REMOVED

python3 -m pip install transformers tqdm beautifulsoup4 huggingface_hub
mkdir -p /workspace/data /workspace/data/encoder_checkpoint

hf download edereynal/financial_prediction \
    documents.pt \
    mlm_encoder/full_model.pt \
    --repo-type dataset \
    --local-dir /workspace/data

mv /workspace/data/mlm_encoder/full_model.pt /workspace/data/encoder_checkpoint/full_model.pt
rmdir /workspace/data/mlm_encoder 2>/dev/null || true
