#!/usr/bin/env bash
# IMU test： test_scores.pkl（ispl_inception · full_w256）
# IMU  CPU ，。
#
# :
#   bash run_imu_test_scores.sh           # xsub + xacq
#   bash run_imu_test_scores.sh xsub      # 
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$ROOT/multimodal/logs"
mkdir -p "$LOG_DIR"

if [[ $# -ge 1 ]]; then
  PROTOCOLS=("$@")
else
  PROTOCOLS=(xsub xacq)
fi

cd "$ROOT"
for proto in "${PROTOCOLS[@]}"; do
  LOG="$LOG_DIR/export_imu_${proto}_$(date +%Y%m%d_%H%M%S).log"
  echo "=== IMU test scores: $proto ===" | tee "$LOG"
  python3 export_test_scores.py --protocol "$proto" --modality imu 2>&1 | tee -a "$LOG"
  echo "Log: $LOG"
done
