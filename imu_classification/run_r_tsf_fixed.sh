#!/usr/bin/env bash
#  r_tsf_net（xsub + xacq）， 2 。
#
# ：
#   cd ${BADMINTONSKILL23_ROOT}/training
#   bash run_r_tsf_fixed.sh              # GPU 0=xsub, 1=xacq
#   GPUS="2 3" bash run_r_tsf_fixed.sh  # 
#
# ：
#   tail -f logs/r_tsf_fixed_<stamp>/xsub_r_tsf_net.log
#   tail -f logs/r_tsf_fixed_<stamp>/xacq_r_tsf_net.log
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/home/ubuntu/users/znc/miniconda3/bin/python3}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/usr/local/cuda-11.8/lib64}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

if [[ -n "${GPUS:-}" ]]; then
  # shellcheck disable=SC2206
  GPU_ARR=($GPUS)
else
  GPU_ARR=(0 1)
fi
GPU_XSUB="${GPU_ARR[0]:-0}"
GPU_XACQ="${GPU_ARR[1]:-${GPU_XSUB}}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="logs/r_tsf_fixed_${STAMP}"
mkdir -p "$LOGDIR" outputs/5node/_broken_r_tsf

# （ results.json ）， 4gpu 
archive_old() {
  local proto="$1"
  shopt -s nullglob
  local dirs=(outputs/5node/${proto}_r_tsf_net_full_w256_*)
  shopt -u nullglob
  for d in "${dirs[@]}"; do
    [[ -d "$d" ]] || continue
    base="$(basename "$d")"
    #  broken 
    [[ "$d" == *"/_broken_r_tsf/"* ]] && continue
    if [[ -f "$d/results.json" ]] || [[ -f "$d/best.keras" ]]; then
      echo "[archive] $d -> outputs/5node/_broken_r_tsf/${base}"
      mv "$d" "outputs/5node/_broken_r_tsf/${base}"
    fi
  done
}

echo "============================================================"
echo " r_tsf_net FIXED  |  save_weights + no restore_best_weights"
echo " GPU xsub=$GPU_XSUB  xacq=$GPU_XACQ"
echo " logdir=$LOGDIR"
echo "============================================================"

archive_old xsub
archive_old xacq

COMMON=(
  --model r_tsf_net
  --win-size 256 --window-mode full
  --epochs 200 --patience 40 --batch-size 64
  --normalize --class-weight
  --lr-schedule cosine_warmup --warmup-epochs 5
)

echo "[cache] ensure 5-node arrays ..."
"$PYTHON" - <<'PY'
from dataset import build_arrays
for p in ("xsub", "xacq"):
    d = build_arrays(p, win_size=256, window_mode="full")
    s = d.meta["shapes"]
    print(f"  {p}: train={s['train']}  val={s['val']}  test={s['test']}  n_ch={d.meta['n_channels']}")
PY

echo "[train] launch xsub on GPU $GPU_XSUB ..."
(
  export CUDA_VISIBLE_DEVICES="$GPU_XSUB"
  "$PYTHON" train_r_tsf_fixed.py --protocol xsub "${COMMON[@]}"
) >"$LOGDIR/xsub_r_tsf_net.log" 2>&1 &
PID_XSUB=$!

echo "[train] launch xacq on GPU $GPU_XACQ ..."
(
  export CUDA_VISIBLE_DEVICES="$GPU_XACQ"
  "$PYTHON" train_r_tsf_fixed.py --protocol xacq "${COMMON[@]}"
) >"$LOGDIR/xacq_r_tsf_net.log" 2>&1 &
PID_XACQ=$!

echo "PIDs: xsub=$PID_XSUB  xacq=$PID_XACQ"
echo "tail -f $LOGDIR/xsub_r_tsf_net.log"
echo "tail -f $LOGDIR/xacq_r_tsf_net.log"

fail=0
wait "$PID_XSUB" || fail=1
wait "$PID_XACQ" || fail=1

echo ""
echo "======== done (fail=$fail) ========"
for proto in xsub xacq; do
  shopt -s nullglob
  dirs=(outputs/5node/${proto}_r_tsf_net_full_w256_*)
  shopt -u nullglob
  latest=""
  for d in "${dirs[@]}"; do
    [[ -f "$d/results.json" ]] && latest="$d"
  done
  if [[ -n "$latest" ]]; then
    "$PYTHON" - <<PY
import json
from pathlib import Path
r = json.loads(Path("$latest/results.json").read_text())
t, v = r.get("test") or {}, r.get("val") or {}
print(f"  {r['protocol']}: test_acc={t.get('accuracy')}  val_acc={v.get('accuracy')}  dir=$latest")
PY
  else
    echo "  $proto: NO results.json"
    fail=1
  fi
done

exit "$fail"
