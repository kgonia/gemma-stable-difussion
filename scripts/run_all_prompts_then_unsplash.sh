#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs/all_prompts_then_unsplash

export WANDB_MODE=online
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}."

uv run --frozen python train.py config_phase0_ella_tsc_all_prompts.json \
  --phases pretrain 2>&1 | tee logs/all_prompts_then_unsplash/phase0.log

uv run --frozen python scripts/finalize_phase0_checkpoint.py \
  config_phase0_ella_tsc_all_prompts.json \
  output_phase0_ella_tsc_all_prompts/ella_connector_clip_pretrain_best.pt \
  output_phase0_ella_tsc_all_prompts/pure_ella_connector_L77.pt \
  2>&1 | tee logs/all_prompts_then_unsplash/phase0_finalize.log

uv run --frozen python train.py config_unsplash_all_p1.json \
  --phases ella 2>&1 | tee logs/all_prompts_then_unsplash/phase1.log
