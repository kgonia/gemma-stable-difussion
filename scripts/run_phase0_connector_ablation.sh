#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs/phase0_connector_ablation

configs=(
  config_phase0_ella_tsc.json
  config_phase0_recursive_y.json
  config_phase0_trm_yz.json
)

failed=0
for config in "${configs[@]}"; do
  name="${config#config_phase0_}"
  name="${name%.json}"
  log="logs/phase0_connector_ablation/${name}.log"
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
