#!/usr/bin/env bash
# Setup script for MLM training on vast.ai.
# Assumes pytorch + cuda are already installed.
set -e

export HF_TOKEN=hf_REMOVED

pip install transformers tqdm beautifulsoup4 huggingface_hub
mkdir -p /workspace/data

hf download edereynal/financial_prediction \
    documents.pt \
    --repo-type dataset \
    --local-dir /workspace/data
