#!/usr/bin/env bash
# BlockGCN native full training (150 epochs, base_lr=0.05, step [120,140]) on Beiyu xsub + xacq.
#
# Usage:
#   bash run_train_blockgcn_native.sh              # both protocols, sequential
#   bash run_train_blockgcn_native.sh xsub         # one protocol
#   CUDA_VISIBLE_DEVICES=0 bash run_train_blockgcn_native.sh
set -euo pipefail

SKELETON_DIR="$(cd "$(dirname "$0")" && pwd)"
BLOCK_ROOT="$SKELETON_DIR/BlockGCN-main"
PYTHON="${PYTHON:-/home/ubuntu/users/znc/miniconda3/envs/vmamba/bin/python}"
LOG_DIR="${LOG_DIR:-$SKELETON_DIR/logs}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -x "$PYTHON" ]]; then
  echo "PYTHON not found: $PYTHON"
  exit 1
fi

if [[ $# -ge 1 && "$1" != --* ]]; then
  PROTOCOLS=("$1")
else
  PROTOCOLS=(xsub xacq)
fi

ensure_npz() {
  local protocol="$1"
  local npz="$SKELETON_DIR/data/blockgcn/beiyu_${protocol}.npz"
  if [[ ! -f "$npz" ]]; then
    echo "ERROR: missing $npz"
    echo "BlockGCN NPZ must exist before native training."
    exit 1
  fi
}

train_protocol() {
  local protocol="$1"
  local config="$BLOCK_ROOT/config/beiyu-${protocol}/joint_native.yaml"
  local work_dir="$SKELETON_DIR/outputs/$protocol/blockgcn_native"
  local stamp log

  ensure_npz "$protocol"

  if [[ ! -f "$config" ]]; then
    echo "Config not found: $config"
    exit 1
  fi

  mkdir -p "$work_dir" "$LOG_DIR"
  # BlockGCN prompts to delete tensorboard runs if the folder exists
  rm -rf "$work_dir/runs"

  stamp="$(date +%Y%m%d_%H%M%S)"
  log="$LOG_DIR/${protocol}_blockgcn_native_${stamp}.log"

  echo ""
  echo "========== BlockGCN native | $protocol | 150 ep | lr=0.05 =========="
  echo "  python : $PYTHON"
  echo "  config : $config"
  echo "  workdir: $work_dir"
  echo "  gpu    : $GPU"
  echo "  log    : $log"

  cd "$BLOCK_ROOT"
  export PYTHONPATH="$BLOCK_ROOT/torchlight:$BLOCK_ROOT:${PYTHONPATH:-}"
  export PYTHONUNBUFFERED=1
  export CUDA_VISIBLE_DEVICES="$GPU"

  "$PYTHON" main.py \
    --config "config/beiyu-${protocol}/joint_native.yaml" \
    --model model.BlockGCN.Model \
    --phase train \
    --device 0 \
    --work-dir "$work_dir" \
    2>&1 | tee "$log"
}

for protocol in "${PROTOCOLS[@]}"; do
  train_protocol "$protocol"
done

echo ""
echo "Done. Outputs: $SKELETON_DIR/outputs/{xsub,xacq}/blockgcn_native/"
