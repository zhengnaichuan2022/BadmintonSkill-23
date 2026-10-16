#!/usr/bin/env bash
# Train all PYSKL skeleton models on GPU 1 (physical card index 1).
# 6 models × 2 protocols (xsub, xacq), with overall progress bar.
#
# Usage:
#   bash run_benchmark_gpu1.sh
#   SKIP_PREPARE=1 bash run_benchmark_gpu1.sh   # default when pickle exists
#   SKIP_TEST=1 bash run_benchmark_gpu1.sh      # train only
set -uo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export NUM_GPUS="${NUM_GPUS:-1}"
export PYSKL_PYTHON="${PYSKL_PYTHON:-/home/ubuntu/users/znc/miniconda3/envs/vmamba/bin/python}"
export SKIP_PREPARE="${SKIP_PREPARE:-1}"
export SKIP_TEST="${SKIP_TEST:-0}"

SKELETON_DIR="$(pwd)"
LOG_DIR="${LOG_DIR:-$SKELETON_DIR/logs/benchmark_gpu1_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
export LOG_DIR

PROTOCOLS=(xsub xacq)
MODELS=(
  stgcnpp
  stgcn
  ctrgcn
  msg3d
  aagcn
  posec3d_joint
)

TOTAL_JOBS=$((${#PROTOCOLS[@]} * ${#MODELS[@]}))
JOB_IDX=0
OK=0
FAIL=0

print_job_header() {
  local protocol="$1"
  local model="$2"
  local idx="$3"
  local total="$4"
  local bar_width=30
  local filled=$((idx * bar_width / total))
  local empty=$((bar_width - filled))
  local bar
  bar="$(printf '%*s' "$filled" '' | tr ' ' '#')$(printf '%*s' "$empty" '' | tr ' ' '-')"
  echo ""
  echo "=============================================="
  echo "   [${bar}] ${idx}/${total}  (ok=${OK} fail=${FAIL})"
  echo "  GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "  : ${protocol} / ${model}"
  echo "=============================================="
}

echo "Skeleton benchmark on GPU ${CUDA_VISIBLE_DEVICES}"
echo "  python : ${PYSKL_PYTHON}"
echo "  logdir : ${LOG_DIR}"
echo "  jobs   : ${TOTAL_JOBS}"
echo "  SKIP_PREPARE=${SKIP_PREPARE} SKIP_TEST=${SKIP_TEST}"
echo "" | tee "$LOG_DIR/master.log"

if [[ "${SKIP_PREPARE}" != "1" ]]; then
  echo "========== PREPARE $(date -Iseconds) ==========" | tee -a "$LOG_DIR/master.log"
  bash run_prepare.sh 2>&1 | tee -a "$LOG_DIR/prepare.log"
else
  echo "========== SKIP PREPARE (pickle/config already exist) ==========" | tee -a "$LOG_DIR/master.log"
  for protocol in "${PROTOCOLS[@]}"; do
    if [[ ! -f "$SKELETON_DIR/annotations/${protocol}/beiyu.pkl" ]]; then
      echo "ERROR: missing $SKELETON_DIR/annotations/${protocol}/beiyu.pkl"
      echo "Run: bash run_prepare.sh"
      exit 1
    fi
  done
fi

for protocol in "${PROTOCOLS[@]}"; do
  for model in "${MODELS[@]}"; do
    JOB_IDX=$((JOB_IDX + 1))
    print_job_header "$protocol" "$model" "$JOB_IDX" "$TOTAL_JOBS"
    echo "========== START ${protocol}_${model} $(date -Iseconds) ==========" | tee -a "$LOG_DIR/master.log"
    JOB_LOG="$LOG_DIR/${protocol}_${model}.log"
    export BENCHMARK_LOG="$JOB_LOG"
    if bash run_train.sh "$protocol" "$model"; then
      unset BENCHMARK_LOG
      OK=$((OK + 1))
      echo "========== DONE  ${protocol}_${model} $(date -Iseconds) ==========" | tee -a "$LOG_DIR/master.log"
      if [[ "${SKIP_TEST}" != "1" ]]; then
        TEST_LOG="$LOG_DIR/${protocol}_${model}_test.log"
        bash run_test.sh "$protocol" "$model" 2>&1 | tee "$TEST_LOG" || true
      fi
    else
      unset BENCHMARK_LOG
      FAIL=$((FAIL + 1))
      echo "========== FAIL  ${protocol}_${model} $(date -Iseconds) ==========" | tee -a "$LOG_DIR/master.log"
    fi
  done
done

echo ""
echo "=============================================="
echo "  Benchmark : ok=${OK} fail=${FAIL} / ${TOTAL_JOBS}"
echo "  : ${LOG_DIR}"
echo "  : ${SKELETON_DIR}/outputs/{xsub,xacq}/"
echo "=============================================="
echo "FINISH ok=${OK} fail=${FAIL}" | tee -a "$LOG_DIR/master.log"
