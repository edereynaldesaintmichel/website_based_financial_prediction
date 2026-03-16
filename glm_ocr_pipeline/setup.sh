#!/bin/bash
# Setup script for GLM-OCR pipeline on a GPU instance (Colab, vast.ai, etc.)
set -e

echo "=== Installing dependencies ==="

# vLLM (nightly required for GLM-OCR support)
pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly

# Transformers from git (GLM-OCR model support)
pip install git+https://github.com/huggingface/transformers.git

# HTML → table screenshots (Playwright + Chromium)
pip install playwright
playwright install --with-deps chromium

# HTML → Markdown conversion + parsing
pip install markdownify beautifulsoup4

# Async HTTP + progress bars
pip install aiohttp tqdm

echo ""
echo "=== Pre-downloading GLM-OCR model ==="
python -c "from huggingface_hub import snapshot_download; snapshot_download('zai-org/GLM-OCR')"

echo ""
echo "=== Setup complete ==="
echo "Start vLLM server:"
echo "  nohup vllm serve zai-org/GLM-OCR --served-model-name glm-ocr --port 8000 \\"
echo "      --gpu-memory-utilization 0.95 --max-num-seqs 128 \\"
echo "      --enable-prefix-caching --enable-chunked-prefill --dtype bfloat16 \\"
echo "      > vllm.log 2>&1 &"
echo ""
echo "Then run:"
echo "  python glm_ocr_pipeline/convert.py <input_dir_or_zip> --no-server --limit 5"
