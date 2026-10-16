#!/usr/bin/env bash
# 5-node IMU recognition benchmark (13 models × xsub/xacq), 4-GPU parallel.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/home/ubuntu/users/znc/miniconda3/bin/python3}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/usr/local/cuda-11.8/lib64}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="logs/full_benchmark_5node_${STAMP}"
mkdir -p "$LOGDIR"
MASTER="$LOGDIR/master.log"

MODELS=(
  deep_conv_lstm ispl_inception cnn_lstm cnn vanilla_lstm bilstm
  stacked_lstm residual_bilstm resnet_se mch_cnn_gru icg_net
  mobile_hart_xs r_tsf_net
)
PROTOCOLS=(xsub xacq)
GPUS=(0 1 2 3)

COMMON=(
  --win-size 256 --window-mode full
  --epochs 200 --patience 40 --batch-size 64
  --normalize --class-weight
  --lr-schedule cosine_warmup --warmup-epochs 5
)

echo "=== 5-node IMU benchmark ===" | tee "$MASTER"
echo "python=$PYTHON" | tee -a "$MASTER"
echo "logdir=$LOGDIR" | tee -a "$MASTER"
echo "models=${MODELS[*]}" | tee -a "$MASTER"

# Build caches first (single process)
echo "========== BUILD CACHE $(date -Iseconds) ==========" | tee -a "$MASTER"
"$PYTHON" - <<'PY' 2>&1 | tee -a "$MASTER"
from dataset import build_arrays
for p in ("xsub", "xacq"):
    d = build_arrays(p, win_size=256, window_mode="full")
    print(p, "shapes", d.meta["shapes"], "n_ch", d.meta["n_channels"])
PY

# Job queue: protocol_model
JOBS=()
for proto in "${PROTOCOLS[@]}"; do
  for m in "${MODELS[@]}"; do
    JOBS+=("${proto}|${m}")
  done
done

declare -A GPU_PID
declare -A GPU_JOB
for g in "${GPUS[@]}"; do
  GPU_PID[$g]=""
  GPU_JOB[$g]=""
done

idx=0
njobs=${#JOBS[@]}

launch() {
  local gpu="$1" proto="$2" model="$3"
  local tag="${proto}_${model}"
  local logfile="$LOGDIR/${tag}.log"
  echo "========== START ${tag} GPU=${gpu} $(date -Iseconds) ==========" | tee -a "$MASTER"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" train.py \
    --model "$model" --protocol "$proto" \
    "${COMMON[@]}" \
    >"$logfile" 2>&1 &
  GPU_PID[$gpu]=$!
  GPU_JOB[$gpu]="$tag"
}

wait_slot() {
  while true; do
    for g in "${GPUS[@]}"; do
      local pid="${GPU_PID[$g]}"
      if [[ -z "$pid" ]]; then
        echo "$g"
        return 0
      fi
      if ! kill -0 "$pid" 2>/dev/null; then
        local rc=0
        wait "$pid" || rc=$?
        local tag="${GPU_JOB[$g]}"
        if [[ $rc -eq 0 ]]; then
          echo "========== DONE  ${tag} GPU=${g} $(date -Iseconds) ==========" | tee -a "$MASTER"
        else
          echo "========== FAIL  ${tag} GPU=${g} rc=${rc} $(date -Iseconds) ==========" | tee -a "$MASTER"
        fi
        GPU_PID[$g]=""
        GPU_JOB[$g]=""
        echo "$g"
        return 0
      fi
    done
    sleep 15
  done
}

while (( idx < njobs )); do
  gpu="$(wait_slot)"
  IFS='|' read -r proto model <<<"${JOBS[$idx]}"
  launch "$gpu" "$proto" "$model"
  idx=$((idx + 1))
done

# drain
for g in "${GPUS[@]}"; do
  pid="${GPU_PID[$g]:-}"
  if [[ -n "$pid" ]]; then
    rc=0
    wait "$pid" || rc=$?
    tag="${GPU_JOB[$g]}"
    if [[ $rc -eq 0 ]]; then
      echo "========== DONE  ${tag} GPU=${g} $(date -Iseconds) ==========" | tee -a "$MASTER"
    else
      echo "========== FAIL  ${tag} GPU=${g} rc=${rc} $(date -Iseconds) ==========" | tee -a "$MASTER"
    fi
  fi
done

# summary
"$PYTHON" - <<PY 2>&1 | tee "$LOGDIR/summary.md"
import json
from pathlib import Path
root = Path("outputs/5node")
rows = []
for d in sorted(root.glob("*_full_w256_*")):
    rj = d / "results.json"
    if not rj.is_file():
        continue
    r = json.loads(rj.read_text())
    t = r.get("test") or {}
    v = r.get("val") or {}
    rows.append((r.get("model"), r.get("protocol"), t.get("accuracy"), t.get("macro_f1"),
                 v.get("accuracy"), v.get("macro_f1"), d.name))
print("| model | protocol | test_acc | test_mf1 | val_acc | val_mf1 | run |")
print("|-------|----------|----------|----------|---------|---------|-----|")
for row in rows:
    print("| {} | {} | {} | {} | {} | {} | {} |".format(
        row[0], row[1],
        f"{row[2]:.4f}" if row[2] is not None else "-",
        f"{row[3]:.4f}" if row[3] is not None else "-",
        f"{row[4]:.4f}" if row[4] is not None else "-",
        f"{row[5]:.4f}" if row[5] is not None else "-",
        row[6],
    ))
print(f"\\nTotal runs with results.json: {len(rows)}")
PY

echo "========== ALL DONE $(date -Iseconds) ==========" | tee -a "$MASTER"
echo "Logs: $LOGDIR"
