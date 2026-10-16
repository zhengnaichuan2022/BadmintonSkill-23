#!/usr/bin/env bash
# Train all PYSKL skeleton models on xsub + xacq (COCO-17, single view).
set -euo pipefail
cd "$(dirname "$0")"

bash run_prepare.sh

PROTOCOLS=(xsub xacq)
MODELS=(
  stgcnpp
  stgcn
  ctrgcn
  msg3d
  aagcn
  posec3d_joint
)

LOG_DIR="${LOG_DIR:-logs/benchmark_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
export LOG_DIR

TOTAL_JOBS=$((${#PROTOCOLS[@]} * ${#MODELS[@]}))
JOB_IDX=0

for protocol in "${PROTOCOLS[@]}"; do
  for model in "${MODELS[@]}"; do
    JOB_IDX=$((JOB_IDX + 1))
    echo ""
    echo "========== [$JOB_IDX/$TOTAL_JOBS] $protocol / $model =========="
    if bash run_train.sh "$protocol" "$model"; then
      bash run_test.sh "$protocol" "$model" || true
    fi
  done
done

echo "Benchmark complete. Logs: $LOG_DIR"
