#!/usr/bin/env bash
# Smoke test: Video scratch × xacq, 10 models, 1 epoch each.
#
# Usage:
#   cd ${BADMINTONSKILL23_ROOT}/video
#   bash run_smoke_scratch_xacq.sh              # 4-GPU parallel
#   CUDA_VISIBLE_DEVICES=0 bash run_smoke_scratch_xacq.sh
#   GPUS=0,1 bash run_smoke_scratch_xacq.sh
#
# Success: train reaches epoch 1 + val epoch 1, no Traceback/OOM.
set -euo pipefail
cd "$(dirname "$0")"

export FROM_SCRATCH=1
export SKIP_EXTRACT_FRAMES=1
export FRAMES_CACHE=frames
export PYTHONUNBUFFERED=1
export MMACTION_PYTHON="${MMACTION_PYTHON:-/home/ubuntu/users/syq/miniconda3/envs/RTMLib/bin/python}"
export MMACTION2_ROOT="${MMACTION2_ROOT:-/home/ubuntu/users/dyc/code/mmaction2}"

IFS=',' read -r -a GPU_LIST <<< "${GPUS:-0,1,2,3}"
NUM_GPUS="${#GPU_LIST[@]}"

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

PROTOCOL=xacq
WORK_TAG=scratch_smoke
#  smoke：50 iter //； SMOKE_FULL_EPOCH=1  1 epoch
SMOKE_ITERS="${SMOKE_ITERS:-20}"
SMOKE_NO_VAL="${SMOKE_NO_VAL:-1}"
SMOKE_FULL_EPOCH="${SMOKE_FULL_EPOCH:-0}"
LOG_DIR="${LOG_DIR:-logs/smoke_scratch_xacq_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.md"

echo "==> Regenerate xacq scratch configs (frames cache)"
python3 generate_configs.py --protocol "$PROTOCOL" --from-scratch

if ! "$MMACTION_PYTHON" check_env.py; then
  echo "Environment check failed"
  exit 1
fi

VIDEO_DIR="$(pwd)"
OK=0
FAIL=0
declare -A RESULT

run_smoke() {
  local gpu="$1" model="$2"
  local config="$VIDEO_DIR/configs/mmaction/$PROTOCOL/$model.py"
  local work_dir="$VIDEO_DIR/outputs/$PROTOCOL/${model}_${WORK_TAG}"
  local smoke_cfg="$work_dir/smoke_cfg.py"
  local log="$LOG_DIR/${PROTOCOL}_${model}.log"

  rm -rf "$work_dir"
  mkdir -p "$work_dir"

  echo "========== START ${PROTOCOL}/${model} GPU${gpu} $(date -Iseconds) ==========" | tee -a "$LOG_DIR/master.log"

  local     train_config="$config"
  if [[ "$SMOKE_FULL_EPOCH" != "1" ]]; then
    export PYTHONPATH="$MMACTION2_ROOT:$VIDEO_DIR:${PYTHONPATH:-}"
    local no_val_flag=()
    [[ "$SMOKE_NO_VAL" == "1" ]] && no_val_flag=(--no-val)
    train_config="$(
      "$MMACTION_PYTHON" "$VIDEO_DIR/prepare_smoke_config.py" \
        "$config" "$smoke_cfg" --iters "$SMOKE_ITERS" "${no_val_flag[@]}"
    )"
  fi

  local cfg_extra=()
  if [[ "$SMOKE_FULL_EPOCH" == "1" ]]; then
    cfg_extra=(
      train_cfg.max_epochs=1
      train_cfg.val_interval=1
      default_hooks.checkpoint.interval=1
    )
  fi

  set +e
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export PYTHONPATH="$MMACTION2_ROOT:$VIDEO_DIR:${PYTHONPATH:-}"
    cd "$MMACTION2_ROOT"
    if [[ "$SMOKE_FULL_EPOCH" == "1" ]]; then
      "$MMACTION_PYTHON" tools/train.py "$train_config" \
        --work-dir "$work_dir" \
        --cfg-options "${cfg_extra[@]}" default_hooks.logger.interval=10 \
        2>&1 | tee "$log"
    else
      "$MMACTION_PYTHON" tools/train.py "$train_config" \
        --work-dir "$work_dir" \
        2>&1 | tee "$log"
    fi
  )
  local rc=${PIPESTATUS[0]}
  set -e

  local status reason
  if [[ "$rc" -ne 0 ]]; then
    status=FAIL
    reason="exit=$rc"
  elif grep -qE "Traceback|CUDA out of memory|RuntimeError|FileNotFoundError|PermissionError" "$log"; then
    status=FAIL
    reason="error in log"
  elif [[ "$SMOKE_FULL_EPOCH" == "1" ]]; then
    if ! grep -q "Epoch(train) \[1\]" "$log"; then
      status=FAIL
      reason="no train ep1"
    elif ! grep -q "Epoch(val) \[1\]" "$log"; then
      status=FAIL
      reason="no val ep1"
    else
      status=OK
      reason="ep1 train+val"
    fi
  elif ! grep -qE "Epoch\(train\)|Iter\(train\)" "$log"; then
    status=FAIL
    reason="no train iter"
  else
    status=OK
    reason="${SMOKE_ITERS} iters"
  fi

  RESULT["$model"]="$status|$reason"
  if [[ "$status" == OK ]]; then
    OK=$((OK + 1))
    echo "========== OK   ${PROTOCOL}/${model} GPU${gpu} $(date -Iseconds) ==========" | tee -a "$LOG_DIR/master.log"
  else
    FAIL=$((FAIL + 1))
    echo "========== FAIL ${PROTOCOL}/${model} GPU${gpu} ($reason) $(date -Iseconds) ==========" | tee -a "$LOG_DIR/master.log"
  fi
}

idx=0
total="${#MODELS[@]}"
while (( idx < total )); do
  pids=()
  names=()
  for ((slot=0; slot<NUM_GPUS && idx<total; slot++, idx++)); do
    model="${MODELS[$idx]}"
    gpu="${GPU_LIST[$slot]}"
    run_smoke "$gpu" "$model" &
    pids+=("$!")
    names+=("$model")
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || true
  done
done

{
  echo "# Video scratch xacq smoke test"
  echo ""
  echo "- time: $(date -Iseconds)"
  echo "- GPUs: ${GPU_LIST[*]}"
  echo "- work_tag: ${WORK_TAG}"
  echo "- smoke_iters: ${SMOKE_ITERS} (SMOKE_FULL_EPOCH=${SMOKE_FULL_EPOCH})"
  echo "- ok/fail: ${OK}/${FAIL}"
  echo ""
  echo "| model | status | note |"
  echo "|---|---|---|"
  for model in "${MODELS[@]}"; do
    IFS='|' read -r st note <<< "${RESULT[$model]:-|FAIL|missing}"
    echo "| $model | $st | $note |"
  done
} | tee "$SUMMARY"

echo ""
echo "=============================================="
echo "  Smoke finished: OK=$OK FAIL=$FAIL"
echo "  summary: $SUMMARY"
echo "=============================================="

[[ "$FAIL" -eq 0 ]]
