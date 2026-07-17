#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 CONFIG_JSON LOG_PATH [GPU_UUID]" >&2
  exit 2
fi

CONFIG_JSON="$1"
LOG_PATH="$2"
EXPECTED_GPU_UUID="${3:-GPU-27ff99d6-d77d-2226-a4b5-7a11b8ef6bdd}"
LOCK="/tmp/hermes-gpu-0.lock"
CUDA_DEVICE="0"

mkdir -p "$(dirname "$LOG_PATH")"

flock -n -E 75 "$LOCK" bash -lc '
  set -euo pipefail
  CONFIG_JSON="$1"
  LOG_PATH="$2"
  EXPECTED_GPU_UUID="$3"
  CUDA_DEVICE="$4"

  export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  echo "GPU queue lease acquired: lock=/tmp/hermes-gpu-0.lock cuda_visible_devices=${CUDA_VISIBLE_DEVICES} expected_uuid=${EXPECTED_GPU_UUID}"
  actual_uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$CUDA_DEVICE" | head -n1 | tr -d "[:space:]")"
  echo "GPU UUID observed: ${actual_uuid}"
  if [[ "${actual_uuid}" != "${EXPECTED_GPU_UUID}" ]]; then
    echo "ERROR: GPU UUID mismatch: expected=${EXPECTED_GPU_UUID} actual=${actual_uuid}" >&2
    exit 77
  fi

  compute_processes="$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader -i "$CUDA_DEVICE" || true)"
  echo "GPU compute processes before launch:"
  if [[ -n "${compute_processes}" ]]; then
    echo "${compute_processes}"
    echo "ERROR: unexpected GPU compute process present; refusing to launch" >&2
    exit 76
  fi
  echo "<none>"

  echo "Working directory: $(pwd)"
  echo "Config: ${CONFIG_JSON}"
  echo "Log: ${LOG_PATH}"
  echo "Command: uv run python scripts/train_longclip_sara.py ${CONFIG_JSON}"

  uv run python scripts/train_longclip_sara.py "${CONFIG_JSON}" 2>&1 | tee "${LOG_PATH}"
' bash "$CONFIG_JSON" "$LOG_PATH" "$EXPECTED_GPU_UUID" "$CUDA_DEVICE"
