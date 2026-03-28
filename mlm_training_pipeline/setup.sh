#!/usr/bin/env bash
# Setup script for vast.ai PyTorch instances.
# Assumes pytorch + cuda are already installed.
set -e

pip install transformers tqdm beautifulsoup4


cd /workspace/website_based_financial_prediction && git pull origin && python -m mlm_training_pipeline.train_mlm_full --data /workspace/data/documents.pt --tokens_per_batch 21248 --lr 6e-5 --number_lr 2e-4 --dtype bf16 --device cuda --compile --resume_from checkpoints/mlm_full/checkpoint_epoch2
