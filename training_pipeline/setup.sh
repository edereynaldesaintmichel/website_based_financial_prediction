#!/usr/bin/env bash
# Setup script for vast.ai PyTorch instances.
# Assumes pytorch + cuda are already installed.
set -e

pip install transformers tqdm beautifulsoup4 gdown

# Download bucketed training data from Google Drive
gdown --folder "https://drive.google.com/drive/folders/12jrUerUuyBJRghg8R_mk31Af2v-jbqVL" --remaining-ok -O /workspace/data/
