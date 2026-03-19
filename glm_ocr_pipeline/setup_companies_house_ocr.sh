#!/bin/bash
# Setup script for the Companies House HTML OCR pipeline on vast.ai.
# vLLM is already installed — this installs missing deps, downloads data,
# and starts the vLLM server once deps are ready.
set -e

INPUT_ZIP="/workspace/companies_house_html.zip"
OUTPUT_JSONL="/workspace/companies_house_html_cleaned_up.jsonl"
GDRIVE_FILE_ID="10oUo0a0HTg0PlU5OIKIUGBS5qUPPkRnn"
GDRIVE_JSONL_FILE_ID="1q_48xszklw1OtURn7DnyBFm7Tq0zmzhp"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Disable tmux on reconnect (fixes broken scroll)
touch ~/.no_auto_tmux


# ── Kill any existing vLLM process ───────────────────────────────────────
EXISTING_VLLM=$(pgrep -f "vllm serve" || true)
if [ -n "$EXISTING_VLLM" ]; then
    echo "Killing existing vLLM process(es): $EXISTING_VLLM"
    kill $EXISTING_VLLM 2>/dev/null
    sleep 2
fi


# ── 1. Install dependencies (vLLM needs these before starting) ───────────
echo "=== [1/3] Installing dependencies ==="
pip install --quiet --upgrade \
    "mistral_common>=1.6.0" \
    aiohttp aiofiles tqdm \
    playwright \
    gdown \
    glmocr
# Force-reinstall transformers from git — pip skips it otherwise if already installed
pip install --quiet --force-reinstall --no-deps \
    git+https://github.com/huggingface/transformers.git
echo "  Packages installed."

# Install Playwright's Chromium browser
python3 -m playwright install --with-deps chromium
echo "  Playwright Chromium installed."


# ── 2. Start downloads in background ─────────────────────────────────────
echo ""
echo "=== [2/3] Downloading input data ==="

if [ -f "$INPUT_ZIP" ]; then
    echo "  $INPUT_ZIP already exists, skipping download."
    DL_PID=""
else
    echo "  Downloading $INPUT_ZIP ..."
    gdown "$GDRIVE_FILE_ID" -O "$INPUT_ZIP" &
    DL_PID=$!
fi

if [ -f "$OUTPUT_JSONL" ]; then
    echo "  $OUTPUT_JSONL already exists, skipping download."
    DL_JSONL_PID=""
else
    echo "  Downloading $OUTPUT_JSONL (WIP checkpoint) ..."
    gdown "$GDRIVE_JSONL_FILE_ID" -O "$OUTPUT_JSONL" &
    DL_JSONL_PID=$!
fi


# ── 3. Start vLLM server (deps are now installed) ────────────────────────
echo ""
echo "=== [3/3] Starting vLLM server ==="
nohup vllm serve zai-org/GLM-OCR \
    --served-model-name glm-ocr \
    --port 8001 \
    --allowed-local-media-path / \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 512 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --dtype bfloat16 \
    --speculative_config '{"method":"mtp","num_speculative_tokens":3}' \
    --disable-log-requests \
    > vllm.log 2>&1 &
VLLM_PID=$!
echo "  vLLM started (PID $VLLM_PID), logging to vllm.log"


# ── Wait for vLLM server to be ready ─────────────────────────────────────
echo ""
echo "=== Waiting for vLLM server to be ready ==="
for i in $(seq 1 600); do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "  ERROR: vLLM exited unexpectedly. Check vllm.log"
        exit 1
    fi
    if curl -sf http://127.0.0.1:8001/health > /dev/null 2>&1; then
        echo "  vLLM ready (${i}s elapsed)"
        break
    fi
    if [ $((i % 30)) -eq 0 ]; then
        echo "  Still waiting... (${i}s elapsed)"
    fi
    sleep 1
    if [ $i -eq 600 ]; then
        echo "  ERROR: vLLM failed to start within 10 minutes. Check vllm.log"
        exit 1
    fi
done


# ── Wait for downloads to finish ─────────────────────────────────────────
if [ -n "$DL_PID" ]; then
    echo "Waiting for input zip download to complete..."
    wait $DL_PID
    if [ $? -ne 0 ]; then
        echo "  ERROR: Input zip download failed"
        exit 1
    fi
    echo "  Input zip download complete."
fi

if [ -n "$DL_JSONL_PID" ]; then
    echo "Waiting for output JSONL download to complete..."
    wait $DL_JSONL_PID
    if [ $? -ne 0 ]; then
        echo "  ERROR: Output JSONL download failed"
        exit 1
    fi
    echo "  Output JSONL download complete."
fi


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
