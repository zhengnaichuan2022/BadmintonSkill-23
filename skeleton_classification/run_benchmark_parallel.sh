#!/usr/bin/env bash
# Multi-GPU parallel skeleton benchmark (PYSKL, 6 models × 2 protocols).
#
# Each job uses one GPU (~2 GB VRAM for GCN models). Schedule in batches of N GPUs.
#
# Usage:
#   bash run_benchmark_parallel.sh
#   GPUS=1,2,3 bash run_benchmark_parallel.sh
#   GPUS=2,3 SKIP_PREPARE=1 bash run_benchmark_parallel.sh
#
# Stop a sequential run_benchmark_gpu1.sh first if it uses the same GPUs.
set -uo pipefail
cd "$(dirname "$0")"

export PYSKL_PYTHON="${PYSKL_PYTHON:-/home/ubuntu/users/znc/miniconda3/envs/vmamba/bin/python}"
export NUM_GPUS="${NUM_GPUS:-1}"
export SKIP_PREPARE="${SKIP_PREPARE:-1}"
export SKIP_TEST="${SKIP_TEST:-0}"

IFS=',' read -r -a GPU_LIST <<< "${GPUS:-2,3}"
NUM_SLOTS="${#GPU_LIST[@]}"

SKELETON_DIR="$(pwd)"
LOG_DIR="${LOG_DIR:-$SKELETON_DIR/logs/benchmark_parallel_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
export LOG_DIR
MASTER="$LOG_DIR/master.log"

PROTOCOLS=(xsub xacq)
MODELS=(
  stgcnpp
  stgcn
  ctrgcn
  msg3d
  aagcn
  posec3d_joint
)

JOBS=()
for protocol in "${PROTOCOLS[@]}"; do
  for model in "${MODELS[@]}"; do
    JOBS+=("${protocol}:${model}")
  done
done

TOTAL_JOBS="${#JOBS[@]}"
OK=0
FAIL=0

echo "Skeleton parallel benchmark"
echo "  python : ${PYSKL_PYTHON}"
echo "  GPUs   : ${GPU_LIST[*]} (${NUM_SLOTS} concurrent)"
echo "  jobs   : ${TOTAL_JOBS}"
echo "  logdir : ${LOG_DIR}"
echo "  SKIP_PREPARE=${SKIP_PREPARE} SKIP_TEST=${SKIP_TEST}"
echo "" | tee "$MASTER"

if [[ "${SKIP_PREPARE}" != "1" ]]; then
  echo "========== PREPARE $(date -Iseconds) ==========" | tee -a "$MASTER"
  bash run_prepare.sh 2>&1 | tee "$LOG_DIR/prepare.log"
else
  echo "========== SKIP PREPARE ==========" | tee -a "$MASTER"
  for protocol in "${PROTOCOLS[@]}"; do
    if [[ ! -f "$SKELETON_DIR/annotations/${protocol}/beiyu.pkl" ]]; then
      echo "ERROR: missing $SKELETON_DIR/annotations/${protocol}/beiyu.pkl"
      exit 1
    fi
  done
fi

run_job() {
  local gpu="$1"
  local protocol="$2"
  local model="$3"
  local job_idx="$4"
  local job_log="$LOG_DIR/${protocol}_${model}.log"

  export CUDA_VISIBLE_DEVICES="$gpu"
  export MASTER_PORT=$((29000 + gpu * 137 + job_idx))
  export BENCHMARK_LOG="$job_log"

  echo "========== START ${protocol}_${model} GPU${gpu} $(date -Iseconds) ==========" | tee -a "$MASTER"
  if bash run_train.sh "$protocol" "$model"; then
    echo "========== DONE  ${protocol}_${model} GPU${gpu} $(date -Iseconds) ==========" | tee -a "$MASTER"
    if [[ "${SKIP_TEST}" != "1" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" MASTER_PORT=$((29500 + gpu * 137 + job_idx)) \
        bash run_test.sh "$protocol" "$model" \
        2>&1 | tee "$LOG_DIR/${protocol}_${model}_test.log" || true
    fi
    return 0
  fi
  echo "========== FAIL  ${protocol}_${model} GPU${gpu} $(date -Iseconds) ==========" | tee -a "$MASTER"
  return 1
}

idx=0
batch_num=0
while (( idx < TOTAL_JOBS )); do
  batch_num=$((batch_num + 1))
  pids=()
  names=()
  gpus_used=()

  for ((slot=0; slot<NUM_SLOTS && idx<TOTAL_JOBS; slot++, idx++)); do
    IFS=':' read -r protocol model <<< "${JOBS[$idx]}"
    gpu="${GPU_LIST[$slot]}"
    run_job "$gpu" "$protocol" "$model" "$idx" &
    pids+=("$!")
    names+=("${protocol}_${model}")
    gpus_used+=("$gpu")
  done

  echo ""
  echo "=============================================="
  echo "  Batch ${batch_num}: waiting for ${#pids[@]} jobs on GPU(s) ${gpus_used[*]}"
  echo "  Progress: $((idx - ${#pids[@]}))-${idx} / ${TOTAL_JOBS}  (ok=${OK} fail=${FAIL})"
  echo "=============================================="

  batch_fail=0
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      OK=$((OK + 1))
    else
      FAIL=$((FAIL + 1))
      batch_fail=1
      echo "Batch member failed: ${names[$i]} (GPU ${gpus_used[$i]})" | tee -a "$MASTER"
    fi
  done

  if (( batch_fail )); then
    echo "Some jobs failed in batch ${batch_num}; continuing..." | tee -a "$MASTER"
  fi
done

echo ""
echo "=============================================="
echo "  Parallel benchmark : ok=${OK} fail=${FAIL} / ${TOTAL_JOBS}"
echo "  : ${LOG_DIR}"
echo "  : ${SKELETON_DIR}/outputs/{xsub,xacq}/"
echo "=============================================="
echo "FINISH ok=${OK} fail=${FAIL}" | tee -a "$MASTER"
