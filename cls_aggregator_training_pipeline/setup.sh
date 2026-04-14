#!/usr/bin/env bash
# Setup for CLS aggregator training + eval on vast.ai.
# Assumes pytorch + cuda are already installed.
#
# Always fetches base files (documents.pt, T5 checkpoint) and any precomputed
# artifacts already present on HF (cls_cache/*, aggregator.pt). Training uploads
# each newly-computed cache shard back to HF automatically.
set -e

export HF_TOKEN=hf_REMOVED

REPO="edereynal/financial_prediction"
DATA_DIR="/workspace/data"

python3 -m pip install transformers tqdm beautifulsoup4 huggingface_hub
mkdir -p "${DATA_DIR}" "${DATA_DIR}/t5_checkpoint" \
         "${DATA_DIR}/aggregator" "${DATA_DIR}/aggregator/cls_cache"

# Core files (required).
hf download "${REPO}" documents.pt t5_cls/full_model.pt \
    --repo-type dataset --local-dir "${DATA_DIR}"

mv "${DATA_DIR}/t5_cls/full_model.pt" "${DATA_DIR}/t5_checkpoint/full_model.pt"
rmdir "${DATA_DIR}/t5_cls" 2>/dev/null || true

# Anything under aggregator/ (caches, previously trained aggregator.pt).
# Tolerate absence — first training run will populate it.
hf download "${REPO}" --include "aggregator/*" \
    --repo-type dataset --local-dir "${DATA_DIR}" \
    || echo "  No aggregator artifacts on HF yet."
