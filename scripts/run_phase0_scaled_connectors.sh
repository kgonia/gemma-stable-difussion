#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs/phase0_scaled

configs=(
  config_phase0_scaled_ella_tsc_b64_200k.json
  config_phase0_scaled_trm_yz_b64_200k.json
)

failed=0
for config in "${configs[@]}"; do
  name="${config#config_phase0_scaled_}"
  name="${name%.json}"
  log="logs/phase0_scaled/${name}.log"
  echo "[$(date --iso-8601=seconds)] starting ${config}"
  if uv run --frozen python train.py "$config" --phases pretrain 2>&1 \
      | tee "$log"; then
    echo "[$(date --iso-8601=seconds)] completed ${config}"
  else
    status=${PIPESTATUS[0]}
    echo "[$(date --iso-8601=seconds)] FAILED ${config} status=${status}"
    failed=1
  fi
done

exit "$failed"
