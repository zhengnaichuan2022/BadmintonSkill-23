#!/usr/bin/env bash
# 4  xsub + xacq  benchmark（ +  pipeline ）。
#
# ：
#   cd ${BADMINTONSKILL23_ROOT}/video
#   bash run_benchmark_4gpu.sh
#
#  GPU （ 0,1,2,3）：
#   GPUS=0,1,2,3 bash run_benchmark_4gpu.sh

set -euo pipefail
cd "$(dirname "$0")"

export FROM_SCRATCH="${FROM_SCRATCH:-1}"
export SKIP_EXTRACT_FRAMES=1
export PYTHONUNBUFFERED=1
export MMACTION_PYTHON="${MMACTION_PYTHON:-/home/ubuntu/users/syq/miniconda3/envs/RTMLib/bin/python}"

IFS=',' read -r -a GPU_LIST <<< "${GPUS:-0,1,2,3}"
NUM_GPUS="${#GPU_LIST[@]}"

PROTOCOLS=(xsub xacq)
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

JOBS=()
for protocol in "${PROTOCOLS[@]}"; do
  for model in "${MODELS[@]}"; do
    JOBS+=("${protocol}:${model}")
  done
done

LOG_DIR="${LOG_DIR:-logs/benchmark_4gpu_$(date +%Y%m%d_%H%M%S)$([ "${FROM_SCRATCH}" = "1" ] && echo _scratch)}"
mkdir -p "$LOG_DIR"
MASTER="$LOG_DIR/master.log"

echo "Prepare configs + annotations (rawframes, from-scratch=${FROM_SCRATCH})"
bash run_prepare.sh 2>&1 | tee "$LOG_DIR/prepare.log"

echo "Benchmark: ${#JOBS[@]} jobs on ${NUM_GPUS} GPUs (${GPU_LIST[*]})" | tee "$MASTER"
echo "Started: $(date -Iseconds)" | tee -a "$MASTER"

run_job() {
  local gpu="$1"
  local protocol="$2"
  local model="$3"
  echo "========== START ${protocol}_${model} GPU${gpu} $(date -Iseconds) ==========" | tee -a "$MASTER"
  if CUDA_VISIBLE_DEVICES="$gpu" bash run_train.sh "$protocol" "$model"; then
    echo "========== DONE  ${protocol}_${model} GPU${gpu} $(date -Iseconds) ==========" | tee -a "$MASTER"
    CUDA_VISIBLE_DEVICES="$gpu" bash run_test.sh "$protocol" "$model" || true
  else
    echo "========== FAIL  ${protocol}_${model} GPU${gpu} $(date -Iseconds) ==========" | tee -a "$MASTER"
    return 1
  fi
}

idx=0
total="${#JOBS[@]}"
while (( idx < total )); do
  pids=()
  names=()
  for ((slot=0; slot<NUM_GPUS && idx<total; slot++, idx++)); do
    IFS=':' read -r protocol model <<< "${JOBS[$idx]}"
    gpu="${GPU_LIST[$slot]}"
    run_job "$gpu" "$protocol" "$model" &
    pids+=("$!")
    names+=("${protocol}_${model}")
  done
  fail=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "Batch member failed: ${names[$i]}" | tee -a "$MASTER"
      fail=1
    fi
  done
  if (( fail )); then
    echo "One or more jobs failed in this batch; continuing..." | tee -a "$MASTER"
  fi
done

python3 summarize_results.py --log-dir "$LOG_DIR" || true
echo "Finished: $(date -Iseconds)" | tee -a "$MASTER"
