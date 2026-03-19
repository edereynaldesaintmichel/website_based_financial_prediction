#!/bin/bash
# Setup script for GLM-OCR pipeline on a vast.ai vLLM template instance.
# vLLM is already installed — this just adds missing deps and downloads the model.
set -e

# Disable tmux on reconnect (fixes broken scroll)
touch ~/.no_auto_tmux

echo "=== Installing dependencies ==="

# Transformers from git (GLM-OCR model type support)
pip install git+https://github.com/huggingface/transformers.git

# mistral_common >= 1.6 required by transformers git for ReasoningEffort import
pip install --upgrade mistral_common

# Async HTTP + file I/O + progress bars
pip install aiohttp aiofiles tqdm

echo ""
echo "=== Downloading input data ==="
pip install gdown -q
gdown 1CbZdwoxdbkm_QxP74VUlx65tp0JzngiO
gdown 10Wr4numQpiW-0fuzfr0LmL4AD-Yv0weA


echo ""
echo "=== Starting vLLM server ==="
nohup vllm serve zai-org/GLM-OCR --served-model-name glm-ocr --port 8000 \
    --gpu-memory-utilization 0.95 --max-num-seqs 512 \
    --enable-prefix-caching --dtype bfloat16 \
    --speculative_config '{"method":"mtp","num_speculative_tokens":3}' \
    --disable-log-requests \
    > vllm.log 2>&1 &
echo "vLLM server started (PID $!), logging to vllm.log"
echo ""
echo "=== Setup complete ==="
echo "Run: python glm_ocr_pipeline/ocr_tables.py <tables.zip>"
