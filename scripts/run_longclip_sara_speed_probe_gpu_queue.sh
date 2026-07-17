#!/usr/bin/env bash
set -euo pipefail

EXPECTED_GPU_UUID="${1:-GPU-27ff99d6-d77d-2226-a4b5-7a11b8ef6bdd}"
LOCK="/tmp/hermes-gpu-0.lock"
CUDA_DEVICE="0"
SUMMARY="logs/queued_longclip_resolution_p4b_speed_probe_summary.tsv"
mkdir -p logs

CONFIGS=(
  "config_longclip_sara_15pct_plus_monet4_resolution_p4b_rope_speed_probe_b8_acc1.json"
  "config_longclip_sara_15pct_plus_monet4_resolution_p4b_rope_speed_probe_b6_acc1.json"
  "config_longclip_sara_15pct_plus_monet4_resolution_p4b_rope_speed_probe_b4_acc2.json"
)
LABELS=(
  "p4b_rope_b8_acc1"
  "p4b_rope_b6_acc1"
  "p4b_rope_b4_acc2_baseline"
)

flock -n -E 75 "$LOCK" bash -lc '
  set -euo pipefail
  EXPECTED_GPU_UUID="$1"
  CUDA_DEVICE="$2"
  SUMMARY="$3"
  shift 3
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

  printf "label\tconfig\tlog\tmonitor\texit_code\telapsed_seconds\tpeak_gpu_mem_mib\tstatus\n" > "$SUMMARY"
  best_success=0

  for pair in "$@"; do
    label="${pair%%::*}"
    config="${pair#*::}"
    log="logs/queued_longclip_resolution_p4b_speed_probe_${label}.log"
    monitor="logs/queued_longclip_resolution_p4b_speed_probe_${label}_gpu_mem.csv"
    rm -f "$log" "$monitor"
    echo "=== speed probe ${label} config=${config} ==="
    echo "timestamp_s,gpu_mem_mib" > "$monitor"
    (
      while true; do
        ts="$(date +%s)"
        mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_DEVICE" | head -n1 | tr -d "[:space:]")"
        echo "${ts},${mem}" >> "$monitor"
        sleep 2
      done
    ) &
    monitor_pid="$!"
    start="$(date +%s)"
    set +e
    uv run python scripts/train_longclip_sara.py "$config" 2>&1 | tee "$log"
    rc="${PIPESTATUS[0]}"
    set -e
    end="$(date +%s)"
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    peak="$(tail -n +2 "$monitor" | cut -d, -f2 | sort -n | tail -n1)"
    if [[ -z "${peak}" ]]; then peak="NA"; fi
    elapsed="$((end - start))"
    status="failed"
    if [[ "$rc" == "0" ]]; then
      status="passed"
      best_success=1
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$label" "$config" "$log" "$monitor" "$rc" "$elapsed" "$peak" "$status" >> "$SUMMARY"
    echo "=== speed probe ${label} exit=${rc} elapsed=${elapsed}s peak_gpu_mem_mib=${peak} ==="
  done

  echo "Speed probe summary: ${SUMMARY}"
  cat "$SUMMARY"
  if [[ "$best_success" != "1" ]]; then
    echo "ERROR: all speed probe candidates failed" >&2
    exit 1
  fi
' bash "$EXPECTED_GPU_UUID" "$CUDA_DEVICE" "$SUMMARY" \
  "${LABELS[0]}::${CONFIGS[0]}" \
  "${LABELS[1]}::${CONFIGS[1]}" \
  "${LABELS[2]}::${CONFIGS[2]}"
