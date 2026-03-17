#!/bin/bash
# Lightweight setup for Google Colab (most deps already installed).
set -e

echo "=== Installing missing dependencies ==="

# vLLM (nightly required for GLM-OCR support) — skip upgrade of existing packages
pip install vllm --extra-index-url https://wheels.vllm.ai/nightly 2>&1 | tail -5

# Transformers from git (GLM-OCR model support)
pip install git+https://github.com/huggingface/transformers.git 2>&1 | tail -5

# Playwright (not on Colab by default)
pip install playwright 2>&1 | tail -3
playwright install --with-deps chromium

# HTML → Markdown conversion (bs4 already on Colab)
pip install glmocr 2>&1 | tail -3

# aiohttp, tqdm — already on Colab, just ensure present
pip install -q aiohttp tqdm

echo ""
echo "=== Pre-downloading GLM-OCR model ==="
python -c "from huggingface_hub import snapshot_download; snapshot_download('zai-org/GLM-OCR')"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Start vLLM server:"
echo "  nohup vllm serve zai-org/GLM-OCR --served-model-name glm-ocr --port 8000 \\"
echo "      --gpu-memory-utilization 0.95 --max-num-seqs 128 \\"
echo "      --enable-prefix-caching --enable-chunked-prefill --dtype bfloat16 \\"
echo "      > vllm.log 2>&1 &"
echo ""
echo "Then run:"
echo "  python glm_ocr_pipeline/convert.py <input_dir_or_zip> --no-server --limit 5"
