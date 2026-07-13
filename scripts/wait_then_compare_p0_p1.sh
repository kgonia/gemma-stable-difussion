#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs/all_prompts_then_unsplash

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while tmux has-session -t ella_unsplash_p1 2>/dev/null; do
  sleep 60
done

checkpoint="output_unsplash_all_p1/ella_connector_frozen_unet.pt"
if [[ ! -f "$checkpoint" ]]; then
  echo "ERROR: Phase 1 session ended without $checkpoint" >&2
  exit 1
fi

uv run --frozen python scripts/generate_p0_p1_comparison.py \
  2>&1 | tee logs/all_prompts_then_unsplash/p0_p1_comparison.log
