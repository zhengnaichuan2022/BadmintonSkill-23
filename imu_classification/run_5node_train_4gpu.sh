#!/usr/bin/env bash
# =============================================================================
# 5  IMU：xsub + xacq，4 （ 1 ）
#
# ：（ results.json）， / 。
#
# ：
#   cd ${BADMINTONSKILL23_ROOT}/training
#   bash run_5node_train_4gpu.sh                 # （）
#   FORCE_ALL=1 bash run_5node_train_4gpu.sh    # 
#   bash run_5node_train_4gpu.sh xsub            # / xsub
#   GPUS="0 1 2 3" bash run_5node_train_4gpu.sh
#
# ：
#   -  20s （GPU /  /  epoch）
#   - ：logs/.../<proto>_<model>.log
#   - ：tail -f logs/.../xsub_ispl_inception.log
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/home/ubuntu/users/znc/miniconda3/bin/python3}"
# ；FORCE_ALL=1 
FORCE_ALL="${FORCE_ALL:-0}"
SKIP_DONE="${SKIP_DONE:-1}"
if [[ "$FORCE_ALL" == "1" ]]; then
  SKIP_DONE=0
fi
REFRESH_SEC="${REFRESH_SEC:-20}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/usr/local/cuda-11.8/lib64}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

# GPU 
if [[ -n "${GPUS:-}" ]]; then
  # shellcheck disable=SC2206
  GPU_ARR=($GPUS)
else
  GPU_ARR=(0 1 2 3)
fi

MODELS=(
  deep_conv_lstm ispl_inception cnn_lstm cnn vanilla_lstm bilstm
  stacked_lstm residual_bilstm resnet_se mch_cnn_gru icg_net
  mobile_hart_xs r_tsf_net
)

if [[ $# -eq 0 ]]; then
  PROTOCOLS=(xsub xacq)
else
  PROTOCOLS=("$@")
fi

COMMON=(
  --win-size 256 --window-mode full
  --epochs 200 --patience 40 --batch-size 64
  --normalize --class-weight
  --lr-schedule cosine_warmup --warmup-epochs 5
)

STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="logs/full_benchmark_5node_4gpu_${STAMP}"
mkdir -p "$LOGDIR" outputs/5node
MASTER="$LOGDIR/master.log"
STATUS="$LOGDIR/status.txt"

JOBS=()
for proto in "${PROTOCOLS[@]}"; do
  for m in "${MODELS[@]}"; do
    JOBS+=("${proto}|${m}")
  done
done

already_done() {
  local proto="$1" model="$2"
  shopt -s nullglob
  local dirs=(outputs/5node/${proto}_${model}_full_w256_*)
  shopt -u nullglob
  for d in "${dirs[@]}"; do
    [[ -f "$d/results.json" ]] && return 0
  done
  return 1
}

# （ results.json）
cleanup_incomplete() {
  local proto="$1" model="$2"
  shopt -s nullglob
  local dirs=(outputs/5node/${proto}_${model}_full_w256_*)
  shopt -u nullglob
  for d in "${dirs[@]}"; do
    if [[ ! -f "$d/results.json" ]]; then
      echo "  : $d" | tee -a "$MASTER"
      rm -rf "$d"
    fi
  done
}

# /；
QUEUE=()
SKIPPED=0
SKIP_LIST=()
RUN_LIST=()
for job in "${JOBS[@]}"; do
  IFS='|' read -r proto model <<<"$job"
  tag="${proto}_${model}"
  if [[ "$SKIP_DONE" == "1" ]] && already_done "$proto" "$model"; then
    SKIPPED=$((SKIPPED + 1))
    SKIP_LIST+=("$tag")
    continue
  fi
  cleanup_incomplete "$proto" "$model"
  QUEUE+=("$job")
  RUN_LIST+=("$tag")
done
TOTAL=${#QUEUE[@]}

echo "============================================================" | tee "$MASTER"
echo " 5-node IMU  |  4-GPU parallel  |  channels=30" | tee -a "$MASTER"
echo " GPUs      : ${GPU_ARR[*]}" | tee -a "$MASTER"
echo " protocols : ${PROTOCOLS[*]}" | tee -a "$MASTER"
echo " mode      : $([[ "$SKIP_DONE" == "1" ]] && echo '/' || echo '')" | tee -a "$MASTER"
echo " to_run    : $TOTAL" | tee -a "$MASTER"
echo " skipped   : $SKIPPED  ( results.json)" | tee -a "$MASTER"
echo " python    : $PYTHON" | tee -a "$MASTER"
echo " logdir    : $LOGDIR" | tee -a "$MASTER"
echo "============================================================" | tee -a "$MASTER"

if [[ ${#SKIP_LIST[@]} -gt 0 ]]; then
  echo " (${#SKIP_LIST[@]}):" | tee -a "$MASTER"
  for t in "${SKIP_LIST[@]}"; do
    echo "  OK   $t" | tee -a "$MASTER"
  done
fi
if [[ ${#RUN_LIST[@]} -gt 0 ]]; then
  echo "/ (${#RUN_LIST[@]}):" | tee -a "$MASTER"
  for t in "${RUN_LIST[@]}"; do
    echo "  RUN  $t" | tee -a "$MASTER"
  done
fi
echo "" | tee -a "$MASTER"

if [[ "$TOTAL" -eq 0 ]]; then
  echo "。" | tee -a "$MASTER"
  exit 0
fi

echo "" | tee -a "$MASTER"
echo "[1/2]  /  5-node cache ..." | tee -a "$MASTER"
"$PYTHON" - <<'PY' 2>&1 | tee -a "$MASTER"
from dataset import build_arrays
for p in ("xsub", "xacq"):
    d = build_arrays(p, win_size=256, window_mode="full")
    s = d.meta["shapes"]
    print(f"  {p}: train={s['train']}  val={s['val']}  test={s['test']}  n_ch={d.meta['n_channels']}")
PY

echo "" | tee -a "$MASTER"
echo "[2/2] 4 （ 1 ）..." | tee -a "$MASTER"
echo "  :  tail -f $LOGDIR/<proto>_<model>.log" | tee -a "$MASTER"
echo "" | tee -a "$MASTER"

# 
declare -A GPU_PID
declare -A GPU_TAG
declare -A GPU_LOG
declare -A GPU_START
for g in "${GPU_ARR[@]}"; do
  GPU_PID[$g]=""
  GPU_TAG[$g]=""
  GPU_LOG[$g]=""
  GPU_START[$g]=""
done

DONE_OK=0
DONE_FAIL=0
QIDX=0

launch_on() {
  local gpu="$1" proto="$2" model="$3"
  local tag="${proto}_${model}"
  local logfile="$LOGDIR/${tag}.log"
  echo "[$(date '+%F %T')] START  GPU=$gpu  $tag" | tee -a "$MASTER"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON" train.py \
      --model "$model" --protocol "$proto" \
      "${COMMON[@]}" \
      >"$logfile" 2>&1
  ) &
  GPU_PID[$gpu]=$!
  GPU_TAG[$gpu]="$tag"
  GPU_LOG[$gpu]="$logfile"
  GPU_START[$gpu]="$(date '+%H:%M:%S')"
}

latest_epoch() {
  local log="$1"
  [[ -f "$log" ]] || { echo "-"; return; }
  #  "Epoch x/200"
  local e
  e=$(grep -oE 'Epoch [0-9]+/200' "$log" 2>/dev/null | tail -1 || true)
  if [[ -n "$e" ]]; then
    echo "$e"
  else
    echo "starting"
  fi
}

print_status() {
  {
    echo ""
    echo "======== $(date '+%F %T') | done=${DONE_OK} fail=${DONE_FAIL} left=$((TOTAL - QIDX)) queued_next=$((TOTAL - QIDX)) ========"
    printf "%-5s %-28s %-12s %-10s %s\n" "GPU" "JOB" "EPOCH" "STARTED" "LOG"
    printf "%-5s %-28s %-12s %-10s %s\n" "---" "---" "-----" "-------" "---"
    for g in "${GPU_ARR[@]}"; do
      local pid="${GPU_PID[$g]}"
      local tag="${GPU_TAG[$g]}"
      local log="${GPU_LOG[$g]}"
      local st="${GPU_START[$g]}"
      if [[ -z "$pid" ]]; then
        printf "%-5s %-28s %-12s %-10s %s\n" "$g" "(idle)" "-" "-" "-"
      else
        local ep
        ep=$(latest_epoch "$log")
        printf "%-5s %-28s %-12s %-10s %s\n" "$g" "${tag:--}" "$ep" "${st:--}" "$(basename "$log")"
      fi
    done
    echo "======================================================================="
  } | tee "$STATUS"
  #  master
  local line="status:"
  for g in "${GPU_ARR[@]}"; do
    line+=" GPU$g=${GPU_TAG[$g]:-idle}"
  done
  echo "[$(date '+%F %T')] $line" >>"$MASTER"
}

# （ 4 ）
for g in "${GPU_ARR[@]}"; do
  if (( QIDX >= TOTAL )); then
    break
  fi
  IFS='|' read -r proto model <<<"${QUEUE[$QIDX]}"
  QIDX=$((QIDX + 1))
  launch_on "$g" "$proto" "$model"
done

print_status
echo " ${REFRESH_SEC}s ；Ctrl+C 。"
echo ""

cleanup() {
  echo "" | tee -a "$MASTER"
  echo "[$(date '+%F %T')] ， GPU ..." | tee -a "$MASTER"
  for g in "${GPU_ARR[@]}"; do
    local pid="${GPU_PID[$g]}"
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
      #  train.py 
      pkill -P "$pid" 2>/dev/null || true
    fi
  done
  wait || true
  exit 130
}
trap cleanup INT TERM

# ：，
while true; do
  alive=0
  for g in "${GPU_ARR[@]}"; do
    pid="${GPU_PID[$g]}"
    [[ -z "$pid" ]] && continue
    if kill -0 "$pid" 2>/dev/null; then
      alive=$((alive + 1))
      continue
    fi
    # 
    rc=0
    wait "$pid" || rc=$?
    tag="${GPU_TAG[$g]}"
    if [[ $rc -eq 0 ]]; then
      echo "[$(date '+%F %T')] OK     GPU=$g  $tag" | tee -a "$MASTER"
      DONE_OK=$((DONE_OK + 1))
    else
      echo "[$(date '+%F %T')] FAIL   GPU=$g  $tag  rc=$rc" | tee -a "$MASTER"
      DONE_FAIL=$((DONE_FAIL + 1))
    fi
    GPU_PID[$g]=""
    GPU_TAG[$g]=""
    GPU_LOG[$g]=""
    GPU_START[$g]=""

    # ，（）
    if (( QIDX < TOTAL )); then
      IFS='|' read -r proto model <<<"${QUEUE[$QIDX]}"
      QIDX=$((QIDX + 1))
      launch_on "$g" "$proto" "$model"
      alive=$((alive + 1))
    fi
  done

  print_status

  # 
  if (( alive == 0 && QIDX >= TOTAL )); then
    break
  fi

  sleep "$REFRESH_SEC"
done

echo "" | tee -a "$MASTER"
echo "============================================================" | tee -a "$MASTER"
echo " : OK=${DONE_OK}  FAIL=${DONE_FAIL}  total=${TOTAL}  skipped=${SKIPPED}" | tee -a "$MASTER"
echo "============================================================" | tee -a "$MASTER"

"$PYTHON" - <<'PY' 2>&1 | tee -a "$MASTER"
import json
from pathlib import Path
root = Path("outputs/5node")
print("\n| model | protocol | test_acc | test_mf1 | val_acc | val_mf1 |")
print("|-------|----------|----------|----------|---------|---------|")
n = 0
for d in sorted(root.glob("*_full_w256_*")):
    rj = d / "results.json"
    if not rj.is_file():
        continue
    r = json.loads(rj.read_text())
    t, v = r.get("test") or {}, r.get("val") or {}
    def fmt(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) else "-"
    print(f"| {r.get('model')} | {r.get('protocol')} | {fmt(t.get('accuracy'))} | {fmt(t.get('macro_f1'))} | {fmt(v.get('accuracy'))} | {fmt(v.get('macro_f1'))} |")
    n += 1
print(f"\nruns with results: {n}")
PY

echo ""
echo ": $LOGDIR"
echo ": $(pwd)/outputs/5node"
echo ":  tail -f $LOGDIR/xsub_ispl_inception.log"
