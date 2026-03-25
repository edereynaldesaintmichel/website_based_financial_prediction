#!/usr/bin/env bash
# Setup script for vast.ai PyTorch instances.
# Assumes pytorch + cuda are already installed.
set -e

python3 -m pip install transformers tqdm beautifulsoup4
