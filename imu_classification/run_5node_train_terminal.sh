#!/usr/bin/env bash
# =============================================================================
# 5  IMU ：xsub + xacq（， Keras ）
#
# 4 ：
#   bash run_5node_train_4gpu.sh
#
# ：
#   cd ${BADMINTONSKILL23_ROOT}/training
#   bash run_5node_train_terminal.sh              #  13 
#   bash run_5node_train_terminal.sh xsub         #  xsub
#   GPU=0 bash run_5node_train_terminal.sh        #  GPU
#   SKIP_DONE=1 bash run_5node_train_terminal.sh  #  results.json  run
#
# ：outputs/5node/
# ：logs/full_benchmark_5node_<>/
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/home/ubuntu/users/znc/miniconda3/bin/python3}"
GPU="${GPU:-0}"
SKIP_DONE="${SKIP_DONE:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/usr/local/cuda-11.8/lib64}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

MODELS=(
  deep_conv_lstm ispl_inception cnn_lstm cnn vanilla_lstm bilstm
  stacked_lstm residual_bilstm resnet_se mch_cnn_gru icg_net
  mobile_hart_xs r_tsf_net
)

# ： = xsub + xacq； xsub / xacq / 
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
LOGDIR="logs/full_benchmark_5node_${STAMP}"
mkdir -p "$LOGDIR" outputs/5node
MASTER="$LOGDIR/master.log"

JOBS=()
for proto in "${PROTOCOLS[@]}"; do
  for m in "${MODELS[@]}"; do
    JOBS+=("${proto}|${m}")
  done
done
TOTAL=${#JOBS[@]}
DONE=0
FAIL=0

already_done() {
  local proto="$1" model="$2"
  # + results.json 
  shopt -s nullglob
  local dirs=(outputs/5node/${proto}_${model}_full_w256_*)
  shopt -u nullglob
  for d in "${dirs[@]}"; do
    if [[ -f "$d/results.json" ]]; then
      return 0
    fi
  done
  return 1
}

echo "============================================================" | tee "$MASTER"
echo " 5-node IMU train  |  channels=30  |  GPU=$GPU" | tee -a "$MASTER"
echo " protocols: ${PROTOCOLS[*]}" | tee -a "$MASTER"
echo " models   : ${#MODELS[@]}  |  jobs: $TOTAL" | tee -a "$MASTER"
echo " python   : $PYTHON" | tee -a "$MASTER"
echo " logdir   : $LOGDIR" | tee -a "$MASTER"
echo " outputs  : outputs/5node/" | tee -a "$MASTER"
echo "============================================================" | tee -a "$MASTER"

echo ""
echo "[1/2]  /  5-node cache ..." | tee -a "$MASTER"
"$PYTHON" - <<'PY' 2>&1 | tee -a "$MASTER"
from dataset import build_arrays
for p in ("xsub", "xacq"):
    d = build_arrays(p, win_size=256, window_mode="full")
    s = d.meta["shapes"]
    print(f"  {p}: train={s['train']}  val={s['val']}  test={s['test']}  n_ch={d.meta['n_channels']}")
PY

echo ""
echo "[2/2] （ Keras ）..." | tee -a "$MASTER"
echo ""

idx=0
for job in "${JOBS[@]}"; do
  idx=$((idx + 1))
  IFS='|' read -r proto model <<<"$job"
  tag="${proto}_${model}"
  logfile="$LOGDIR/${tag}.log"

  echo "------------------------------------------------------------" | tee -a "$MASTER"
  echo " [${idx}/${TOTAL}] ${tag}   $(date '+%F %T')" | tee -a "$MASTER"
  echo "------------------------------------------------------------" | tee -a "$MASTER"

  if [[ "$SKIP_DONE" == "1" ]] && already_done "$proto" "$model"; then
    echo "  SKIP ( results.json)" | tee -a "$MASTER"
    DONE=$((DONE + 1))
    continue
  fi

  # ；Keras 
  set +e
  "$PYTHON" train.py \
    --model "$model" --protocol "$proto" \
    "${COMMON[@]}" \
    2>&1 | tee "$logfile"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ $rc -eq 0 ]]; then
    echo "  OK  ${tag}  $(date '+%F %T')" | tee -a "$MASTER"
    DONE=$((DONE + 1))
  else
    echo "  FAIL ${tag}  rc=${rc}  $(date '+%F %T')" | tee -a "$MASTER"
    FAIL=$((FAIL + 1))
  fi
  echo "" | tee -a "$MASTER"
done

echo "============================================================" | tee -a "$MASTER"
echo " : OK=${DONE}  FAIL=${FAIL}  total=${TOTAL}" | tee -a "$MASTER"
echo "============================================================" | tee -a "$MASTER"

# 
"$PYTHON" - <<'PY' 2>&1 | tee -a "$MASTER"
import json
from pathlib import Path

root = Path("outputs/5node")
print("\n| model | protocol | test_acc | test_mf1 | val_acc | val_mf1 |")
print("|-------|----------|----------|----------|---------|---------|")
rows = []
for d in sorted(root.glob("*_full_w256_*")):
    rj = d / "results.json"
    if not rj.is_file():
        continue
    r = json.loads(rj.read_text())
    t, v = r.get("test") or {}, r.get("val") or {}
    def fmt(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) else "-"
    print(f"| {r.get('model')} | {r.get('protocol')} | {fmt(t.get('accuracy'))} | {fmt(t.get('macro_f1'))} | {fmt(v.get('accuracy'))} | {fmt(v.get('macro_f1'))} |")
    rows.append(d.name)
print(f"\nruns with results: {len(rows)}")
PY

echo ""
echo ": $LOGDIR"
echo ": $(pwd)/outputs/5node"
