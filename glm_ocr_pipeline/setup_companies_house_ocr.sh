#!/bin/bash
# Setup script for the Companies House HTML OCR pipeline on vast.ai.
# vLLM is already installed — this installs missing deps, downloads data,
# and starts the vLLM server concurrently.
set -e

INPUT_ZIP="/workspace/companies_house_html.zip"
OUTPUT_JSONL="/workspace/companies_house_html_cleaned_up.jsonl"
GDRIVE_FILE_ID="10oUo0a0HTg0PlU5OIKIUGBS5qUPPkRnn"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Disable tmux on reconnect (fixes broken scroll)
touch ~/.no_auto_tmux


# ── 1. Start vLLM server immediately ─────────────────────────────────────
echo "=== [1/3] Starting vLLM server ==="
nohup vllm serve zai-org/GLM-OCR \
    --served-model-name glm-ocr \
    --port 8000 \
    --allowed-local-media-path / \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 512 \
    --enable-prefix-caching \
    --dtype bfloat16 \
    --speculative_config '{"method":"mtp","num_speculative_tokens":3}' \
    --disable-log-requests \
    > vllm.log 2>&1 &
VLLM_PID=$!
echo "  vLLM started (PID $VLLM_PID), logging to vllm.log"


# ── 2. pip install deps in background (while vLLM warms up) ──────────────
echo ""
echo "=== [2/3] Installing dependencies ==="
pip install --quiet \
    git+https://github.com/huggingface/transformers.git \
    mistral_common \
    aiohttp aiofiles tqdm \
    playwright \
    gdown &
PIP_PID=$!


# ── 3. Wait for pip, then download data ──────────────────────────────────
echo ""
echo "=== [3/3] Downloading input data (waiting for pip first) ==="
wait $PIP_PID
if [ $? -ne 0 ]; then
    echo "  ERROR: pip install failed"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi
echo "  Packages installed."

# Install Playwright's Chromium browser
python3 -m playwright install --with-deps chromium
echo "  Playwright Chromium installed."

# Start download in background (runs while we wait for vLLM)
echo "  Downloading $INPUT_ZIP ..."
gdown "$GDRIVE_FILE_ID" -O "$INPUT_ZIP" &
DL_PID=$!


# ── Wait for vLLM server to be ready ─────────────────────────────────────
echo ""
echo "=== Waiting for vLLM server to be ready ==="
for i in $(seq 1 600); do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "  ERROR: vLLM exited unexpectedly. Check vllm.log"
        kill $DL_PID 2>/dev/null
        exit 1
    fi
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "  vLLM ready (${i}s elapsed)"
        break
    fi
    if [ $((i % 30)) -eq 0 ]; then
        echo "  Still waiting... (${i}s elapsed)"
    fi
    sleep 1
    if [ $i -eq 600 ]; then
        echo "  ERROR: vLLM failed to start within 10 minutes. Check vllm.log"
        kill $DL_PID 2>/dev/null
        exit 1
    fi
done

# ── Wait for download to finish ───────────────────────────────────────────
echo "Waiting for download to complete..."
wait $DL_PID
if [ $? -ne 0 ]; then
    echo "  ERROR: Download failed"
    exit 1
fi
echo "  Download complete."


# ── Done ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  SETUP COMPLETE"
echo "  vLLM server is running · data downloaded · deps installed"
echo "============================================================"
echo ""
echo "Run the pipeline with:"
echo ""
echo "  python3 $SCRIPT_DIR/convert.py $INPUT_ZIP --output $OUTPUT_JSONL"
echo ""
