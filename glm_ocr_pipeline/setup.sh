#!/bin/bash
# Setup script for GLM-OCR pipeline on a vast.ai vLLM template instance.
# vLLM is already installed — this just adds missing deps and downloads the model.
set -e

# Disable tmux on reconnect (fixes broken scroll)
touch ~/.no_auto_tmux

echo "=== Installing dependencies ==="

# Transformers from git (GLM-OCR model type support)
pip install git+https://github.com/huggingface/transformers.git

# Async HTTP + file I/O + progress bars
pip install aiohttp aiofiles tqdm

echo ""
echo "=== Pre-downloading GLM-OCR model ==="
python -c "from huggingface_hub import snapshot_download; snapshot_download('zai-org/GLM-OCR')"

echo ""
echo "=== Setup complete ==="
echo "Start vLLM server:"
echo "  nohup vllm serve zai-org/GLM-OCR --served-model-name glm-ocr --port 8000 \\"
echo "      --gpu-memory-utilization 0.95 --max-num-seqs 512 \\"
echo "      --enable-prefix-caching --enable-chunked-prefill --dtype bfloat16 \\"
echo "      --speculative_config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}' \\"
echo "      > vllm.log 2>&1 &"
echo ""
echo "Then run:"
echo "  python glm_ocr_pipeline/ocr_tables.py <tables.zip>"
