#!/usr/bin/env bash
# ：GPU0 → xsub，GPU1 → xacq， Video test_scores.pkl
#
# :
#   bash run_video_test_scores_2gpu.sh
#   bash run_video_test_scores_2gpu.sh --then-ensemble   #  multimodal_ensemble.py
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
THEN_ENSEMBLE=0
if [[ "${1:-}" == "--then-ensemble" ]]; then
  THEN_ENSEMBLE=1
fi

export MMACTION_PYTHON="${MMACTION_PYTHON:-/home/ubuntu/users/syq/miniconda3/envs/RTMLib/bin/python}"
export MMACTION2_ROOT="${MMACTION2_ROOT:-/home/ubuntu/users/dyc/code/mmaction2}"
export PYTHONPATH="${MMACTION2_ROOT}:${ROOT}/video:${PYTHONPATH:-}"

LOG_DIR="$ROOT/multimodal/logs"
mkdir -p "$LOG_DIR"
MASTER="$LOG_DIR/export_video_2gpu_$(date +%Y%m%d_%H%M%S).log"

{
  echo "=== Video test_scores 2-GPU parallel ==="
  echo "GPU0 -> xsub"
  echo "GPU1 -> xacq"
  echo "MMACTION_PYTHON=$MMACTION_PYTHON"
  echo "MMACTION2_ROOT=$MMACTION2_ROOT"
  echo "Started: $(date -Iseconds)"
} | tee "$MASTER"

run_proto() {
  local gpu="$1" proto="$2"
  echo "-------- START $proto GPU$gpu $(date -Iseconds) --------" | tee -a "$MASTER"
  if bash "$ROOT/run_video_test_scores.sh" "$proto" "$gpu"; then
    echo "-------- DONE  $proto GPU$gpu $(date -Iseconds) --------" | tee -a "$MASTER"
    return 0
  fi
  echo "-------- FAIL  $proto GPU$gpu $(date -Iseconds) --------" | tee -a "$MASTER"
  return 1
}

ec=0
run_proto 0 xsub &
pid0=$!
run_proto 1 xacq &
pid1=$!

wait "$pid0" || ec=1
wait "$pid1" || ec=1

if [[ "$ec" -ne 0 ]]; then
  echo "FAILED: see $LOG_DIR/export_video_*_gpu*.log" | tee -a "$MASTER"
  exit 1
fi

echo "Both protocols exported OK." | tee -a "$MASTER"

if [[ "$THEN_ENSEMBLE" -eq 1 ]]; then
  echo "--- multimodal_ensemble.py ---" | tee -a "$MASTER"
  cd "$ROOT"
  python3 multimodal_ensemble.py 2>&1 | tee -a "$MASTER"
fi

echo "Finished: $(date -Iseconds)" | tee -a "$MASTER"
echo "Master log: $MASTER"
