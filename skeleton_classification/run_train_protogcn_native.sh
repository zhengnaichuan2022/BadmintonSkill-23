#!/usr/bin/env bash
# ProtoGCN native full training (150 epochs, lr=0.025) on Beiyu xsub + xacq.
#
# Usage:
#   bash run_train_protogcn_native.sh              # both protocols, sequential
#   bash run_train_protogcn_native.sh xsub         # one protocol
#   CUDA_VISIBLE_DEVICES=0 bash run_train_protogcn_native.sh
set -euo pipefail

SKELETON_DIR="$(cd "$(dirname "$0")" && pwd)"
PROTOGCN_ROOT="$SKELETON_DIR/ProtoGCN-main"
PYTHON="${PYTHON:-/home/ubuntu/users/znc/miniconda3/envs/vmamba/bin/python}"
NUM_GPUS="${NUM_GPUS:-1}"
LOG_DIR="${LOG_DIR:-$SKELETON_DIR/logs}"

if [[ ! -x "$PYTHON" ]]; then
  echo "PYTHON not found: $PYTHON"
  exit 1
fi

if [[ $# -ge 1 && "$1" != --* ]]; then
  PROTOCOLS=("$1")
else
  PROTOCOLS=(xsub xacq)
fi

ensure_data() {
  local protocol="$1"
  local pkl="$SKELETON_DIR/annotations/$protocol/beiyu.pkl"
  if [[ ! -f "$pkl" ]]; then
    echo "Missing $pkl — running prepare..."
    bash "$SKELETON_DIR/run_prepare.sh" --protocol "$protocol"
  fi
}

train_protocol() {
  local protocol="$1"
  local config="$SKELETON_DIR/configs/protogcn/$protocol/protogcn_native.py"
  local work_dir="$SKELETON_DIR/outputs/$protocol/protogcn_native"
  local stamp log master_port

  ensure_data "$protocol"

  if [[ ! -f "$config" ]]; then
    echo "Config not found: $config"
    exit 1
  fi

  mkdir -p "$work_dir" "$LOG_DIR"
  stamp="$(date +%Y%m%d_%H%M%S)"
  log="$LOG_DIR/${protocol}_protogcn_native_${stamp}.log"
  master_port="${MASTER_PORT:-$((12000 + RANDOM % 20000))}"

  echo ""
  echo "========== ProtoGCN native | $protocol | 150 ep | lr=0.025 =========="
  echo "  python : $PYTHON"
  echo "  config : $config"
  echo "  workdir: $work_dir"
  echo "  gpus   : $NUM_GPUS"
  echo "  log    : $log"

  cd "$PROTOGCN_ROOT"
  export SKELETON_DIR="$SKELETON_DIR"
  export PYTHONPATH="$PROTOGCN_ROOT:${PYTHONPATH:-}"
  export PYTHONUNBUFFERED=1
  export MKL_SERVICE_FORCE_INTEL=1

  "$PYTHON" -m torch.distributed.run \
    --nproc_per_node="$NUM_GPUS" --master_port="$master_port" \
    tools/train.py "$config" \
    --launcher pytorch --validate --test-last --test-best \
    2>&1 | tee "$log"
}

for protocol in "${PROTOCOLS[@]}"; do
  train_protocol "$protocol"
done

echo ""
echo "Done. Outputs: $SKELETON_DIR/outputs/{xsub,xacq}/protogcn_native/"
