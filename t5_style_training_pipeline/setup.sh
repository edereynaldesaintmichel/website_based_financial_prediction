#!/usr/bin/env bash
# Setup script for vast.ai PyTorch instances.
# Assumes pytorch + cuda are already installed.
set -e

python -m pip install transformers tqdm beautifulsoup4
