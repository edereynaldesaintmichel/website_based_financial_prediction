#!/bin/bash
# Setup script for GLM-OCR pipeline on a GPU instance (Colab, vast.ai, etc.)
set -e

echo "=== Installing dependencies ==="

# vLLM (nightly required for GLM-OCR support)
pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly

# Transformers from git (GLM-OCR model support)
pip install git+https://github.com/huggingface/transformers.git

# HTML → PDF (Playwright + Chromium)
pip install playwright
playwright install --with-deps chromium

# PDF → images + async HTTP
pip install pymupdf aiohttp

echo ""
echo "=== Pre-downloading GLM-OCR model ==="
python -c "from huggingface_hub import snapshot_download; snapshot_download('zai-org/GLM-OCR')"

echo ""
echo "=== Setup complete ==="
echo "Next: python glm_ocr_pipeline/convert.py <input_dir_or_zip> --limit 5"
