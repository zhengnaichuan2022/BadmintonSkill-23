#!/usr/bin/env bash
# Video scratch × xacq： 10  +  test。
#  50 epoch ； TEST_ONLY=1  test。
#
# Usage:
#   cd ${BADMINTONSKILL23_ROOT}/video
#   bash run_benchmark_scratch_xacq.sh --status
#   CUDA_VISIBLE_DEVICES=0 bash run_benchmark_scratch_xacq.sh
#   GPUS=0,1,2,3 bash run_benchmark_scratch_xacq.sh
#   TEST_ONLY=1 bash run_benchmark_scratch_xacq.sh
#   SKIP_DONE=0 bash run_benchmark_scratch_xacq.sh   # 
set -euo pipefail
cd "$(dirname "$0")"

export FROM_SCRATCH=1
export SKIP_EXTRACT_FRAMES=1
export FRAMES_CACHE=frames
export PYTHONUNBUFFERED=1
export MMACTION_PYTHON="${MMACTION_PYTHON:-/home/ubuntu/users/syq/miniconda3/envs/RTMLib/bin/python}"
export MMACTION2_ROOT="${MMACTION2_ROOT:-/home/ubuntu/users/dyc/code/mmaction2}"

PROTOCOL=xacq
SKIP_DONE="${SKIP_DONE:-1}"
TEST_ONLY="${TEST_ONLY:-0}"
PYTHON="${PYTHON:-python3}"

MODELS=(
  tsn_badminton_r50
  tsm_badminton_r50
  slowfast_badminton_r50
  c3d_badminton
  i3d_badminton_r50
  r2plus1d_badminton_r34
  x3d_badminton_m
  tin_badminton_r50
  trn_badminton_r50
  tanet_badminton_r50
)

IFS=',' read -r -a GPU_LIST <<< "${GPUS:-0}"
NUM_GPUS="${#GPU_LIST[@]}"

if [[ "${1:-}" == "--status" ]]; then
  echo "==> Train status"
  "$PYTHON" benchmark_scratch_jobs.py --protocol "$PROTOCOL"
  echo ""
  echo "==> Need test (train done, no test_results.json)"
  "$PYTHON" benchmark_scratch_jobs.py --protocol "$PROTOCOL" --need-test
  exit 0
fi

echo "==> Prepare xacq scratch configs (frames cache)"
python3 generate_configs.py --protocol "$PROTOCOL" --from-scratch

LOG_DIR="${LOG_DIR:-logs/benchmark_scratch_xacq_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
{
  echo "Benchmark scratch xacq: FRAMES_CACHE=$FRAMES_CACHE SKIP_DONE=$SKIP_DONE TEST_ONLY=$TEST_ONLY"
  echo "GPUs: ${GPU_LIST[*]}"
  echo "Started: $(date -Iseconds)"
} | tee "$LOG_DIR/master.log"

train_done() {
  "$PYTHON" -c "
from benchmark_scratch_jobs import job_info
import sys
sys.exit(0 if job_info('$PROTOCOL','$1')['done'] else 1)
"
}

need_test() {
  "$PYTHON" -c "
import sys
from benchmark_scratch_jobs import job_info
j = job_info('$PROTOCOL','$1')
sys.exit(0 if j['done'] and not j['has_test'] else 1)
"
}

run_train() {
  local gpu="$1" model="$2"
  local log="$LOG_DIR/${PROTOCOL}_${model}.log"
  echo "========== TRAIN ${PROTOCOL}/${model} GPU${gpu} $(date -Iseconds) ==========" | tee -a "$LOG_DIR/master.log"
  CUDA_VISIBLE_DEVICES="$gpu" bash run_train.sh "$PROTOCOL" "$model" 2>&1 | tee "$log"
}

run_test() {
  local gpu="$1" model="$2"
  echo "========== TEST  ${PROTOCOL}/${model} GPU${gpu} $(date -Iseconds) ==========" | tee -a "$LOG_DIR/master.log"
  CUDA_VISIBLE_DEVICES="$gpu" bash run_test.sh "$PROTOCOL" "$model" 2>&1 | tee -a "$LOG_DIR/${PROTOCOL}_${model}.log"
}

run_job() {
  local gpu="$1" model="$2"
  if [[ "$TEST_ONLY" == "1" ]]; then
    if need_test "$model"; then
      run_test "$gpu" "$model" || true
    else
      echo "========== SKIP ${PROTOCOL}/${model} (test exists or train incomplete) ==========" | tee -a "$LOG_DIR/master.log"
    fi
    return 0
  fi

  if [[ "$SKIP_DONE" == "1" ]] && train_done "$model"; then
    echo "========== SKIP TRAIN ${PROTOCOL}/${model} (50 ep done) ==========" | tee -a "$LOG_DIR/master.log"
  else
    run_train "$gpu" "$model" || return 1
  fi
  run_test "$gpu" "$model" || true
}

TOTAL="${#MODELS[@]}"
idx=0
OK=0
FAIL=0
SKIP=0

while (( idx < TOTAL )); do
  pids=()
  names=()
  for ((slot=0; slot<NUM_GPUS && idx<TOTAL; slot++, idx++)); do
    model="${MODELS[$idx]}"
    gpu="${GPU_LIST[$slot]}"
    (
      if run_job "$gpu" "$model"; then
        exit 0
      else
        exit 1
      fi
    ) &
    pids+=("$!")
    names+=("$model")
  done
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      OK=$((OK + 1))
    else
      FAIL=$((FAIL + 1))
      echo "FAILED: ${names[$i]}" | tee -a "$LOG_DIR/master.log"
    fi
  done
done

"$PYTHON" summarize_results.py --log-dir "$LOG_DIR" 2>/dev/null || true
echo ""
echo "=============================================="
echo "  scratch xacq benchmark finished"
echo "  ok=$OK fail=$FAIL  logs: $LOG_DIR"
echo "=============================================="
echo "FINISH ok=$OK fail=$FAIL" | tee -a "$LOG_DIR/master.log"
