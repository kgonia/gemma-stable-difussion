#!/usr/bin/env bash
set -e

cd /gemma-stable-difussion

# Source env secrets
source .env_wandb 2>/dev/null || true

# Network volume paths
export HF_HOME=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache/hub
export HUGGINGFACE_HUB_CACHE=/workspace/hf_cache/hub
export HF_DATASETS_CACHE=/workspace/hf_datasets

echo "=== Long 256-token SaRA Training Start ==="
echo "Date: $(date)"
echo "Output: /workspace/output"
echo "WandB: enabled ($([ -n \"$WANDB_API_KEY\" ] && echo 'yes' || echo 'no'))"
echo "Config: config_long_256.json"
echo ""

uv run python train.py config_long_256.json 2>&1

echo ""
echo "=== Training Complete ==="
echo "Date: $(date)"
