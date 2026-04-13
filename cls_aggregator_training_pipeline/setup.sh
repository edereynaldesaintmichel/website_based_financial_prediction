#!/usr/bin/env bash
# Setup script for CLS aggregator training + eval on vast.ai.
# Assumes pytorch + cuda are already installed.
# Usage:
#   bash cls_aggregator_training_pipeline/setup.sh          # training (no aggregator ckpt)
#   bash cls_aggregator_training_pipeline/setup.sh eval     # eval (also fetches aggregator.pt)
set -e

export HF_TOKEN=hf_REMOVED

MODE="${1:-train}"

python3 -m pip install transformers tqdm beautifulsoup4 huggingface_hub
mkdir -p /workspace/data /workspace/data/t5_checkpoint /workspace/data/aggregator

FILES=(documents.pt t5_cls/full_model.pt)
if [ "$MODE" = "eval" ]; then
    FILES+=(aggregator/aggregator.pt)
fi

hf download edereynal/financial_prediction \
    "${FILES[@]}" \
    --repo-type dataset \
    --local-dir /workspace/data

# Flatten t5_cls/ into the path expected by deploy docs.
mv /workspace/data/t5_cls/full_model.pt /workspace/data/t5_checkpoint/full_model.pt
rmdir /workspace/data/t5_cls 2>/dev/null || true
