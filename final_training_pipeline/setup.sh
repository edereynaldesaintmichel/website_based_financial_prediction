#!/usr/bin/env bash
# Setup script for growth predictor training on vast.ai.
# Assumes pytorch + cuda are already installed.
set -e

export HF_TOKEN=hf_REMOVED

python3 -m pip install transformers tqdm beautifulsoup4 huggingface_hub
mkdir -p /workspace/data /workspace/data/t5_checkpoint /workspace/data/aggregator_checkpoint

hf download edereynal/financial_prediction \
    documents.pt \
    growth_rates.json \
    t5_cls/full_model.pt \
    aggregator/aggregator.pt \
    --repo-type dataset \
    --local-dir /workspace/data

mv /workspace/data/t5_cls/full_model.pt /workspace/data/t5_checkpoint/full_model.pt
mv /workspace/data/aggregator/aggregator.pt /workspace/data/aggregator_checkpoint/aggregator.pt
rmdir /workspace/data/t5_cls /workspace/data/aggregator 2>/dev/null || true
