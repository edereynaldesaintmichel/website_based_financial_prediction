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

# GLM-OCR SDK (layout detection + OCR pipeline)
pip install glmocr 2>&1 | tail -3

# aiohttp, pymupdf, tqdm — already on Colab, just ensure present
pip install -q aiohttp pymupdf tqdm

echo ""
echo "=== Pre-downloading GLM-OCR model ==="
python -c "from huggingface_hub import snapshot_download; snapshot_download('zai-org/GLM-OCR')"

echo ""
echo "=== Setup complete ==="
