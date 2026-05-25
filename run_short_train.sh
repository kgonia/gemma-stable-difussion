#!/usr/bin/env bash
set -e

cd /gemma-stable-difussion

# Network volume paths
export HF_HOME=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache/hub
export HUGGINGFACE_HUB_CACHE=/workspace/hf_cache/hub
export HF_DATASETS_CACHE=/workspace/hf_datasets

echo "=== Short Training Start ==="
echo "Date: $(date)"
echo "Output: /workspace/output"
echo "HF cache: /workspace/hf_cache"
echo "Datasets cache: /workspace/hf_datasets"
echo ""

uv run python train.py config.json 2>&1

echo ""
echo "=== Training Complete ==="
echo "Date: $(date)"
