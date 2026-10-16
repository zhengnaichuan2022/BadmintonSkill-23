#!/usr/bin/env bash
# CoM-M2U pipeline: export offline logits → fine-tune RGB/IMU/Skeleton students.
# Edit PYTHON env vars below for your machine.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HERE:${PYTHONPATH:-}"
export BADMINTONSKILL23_PROJECT_ROOT="${BADMINTONSKILL23_PROJECT_ROOT:-example/project}"
export BADMINTONSKILL23_COM_M2U_OUT="${BADMINTONSKILL23_COM_M2U_OUT:-example/work/com_m2u}"

TF_PY="${TF_PYTHON:-python}"
MA_PY="${MMACTION_PYTHON:-python}"
SK_PY="${PYSKL_PYTHON:-python}"

PROTO=${1:-both}   # xsub | xacq | both
STAGE=${2:-all}    # export | finetune | all
OUT="$BADMINTONSKILL23_COM_M2U_OUT"
LOG="$OUT/logs"
mkdir -p "$LOG"

export_one() {
  local proto=$1
  echo "===== EXPORT $proto ====="
  CUDA_VISIBLE_DEVICES=0 $TF_PY -m com_m2u.export_logits --protocol "$proto" --modality imu \
    2>&1 | tee "$LOG/export_imu_${proto}.log"

  CUDA_VISIBLE_DEVICES=0 $MA_PY -m com_m2u.export_logits --protocol "$proto" --modality rgb --gpu 0 \
    >"$LOG/export_rgb_${proto}.log" 2>&1 &
  local p_rgb=$!

  CUDA_VISIBLE_DEVICES=1 $SK_PY -m com_m2u.export_logits --protocol "$proto" --modality skel_streams --stream j --gpu 0 \
    >"$LOG/export_sk_j_${proto}.log" 2>&1 &
  CUDA_VISIBLE_DEVICES=2 $SK_PY -m com_m2u.export_logits --protocol "$proto" --modality skel_streams --stream b --gpu 0 \
    >"$LOG/export_sk_b_${proto}.log" 2>&1 &
  CUDA_VISIBLE_DEVICES=3 $SK_PY -m com_m2u.export_logits --protocol "$proto" --modality skel_streams --stream jm --gpu 0 \
    >"$LOG/export_sk_jm_${proto}.log" 2>&1 &
  wait $p_rgb

  CUDA_VISIBLE_DEVICES=1 $SK_PY -m com_m2u.export_logits --protocol "$proto" --modality skel_streams --stream bm --gpu 0 \
    2>&1 | tee "$LOG/export_sk_bm_${proto}.log"
  $TF_PY -m com_m2u.export_logits --protocol "$proto" --modality fuse_skel \
    2>&1 | tee "$LOG/export_sk_fuse_${proto}.log"
}

finetune_one() {
  local proto=$1
  echo "===== FINETUNE $proto ====="
  CUDA_VISIBLE_DEVICES=2 $TF_PY -m com_m2u.finetune_imu_simple --protocol "$proto" \
    2>&1 | tee "$LOG/ft_imu_${proto}.log"
  CUDA_VISIBLE_DEVICES=0 $MA_PY -m com_m2u.finetune_rgb_simple --protocol "$proto" --device cuda:0 \
    2>&1 | tee "$LOG/ft_rgb_${proto}.log"
  CUDA_VISIBLE_DEVICES=1 $SK_PY -m com_m2u.finetune_skeleton_train --protocol "$proto" --stream j --device cuda:0 \
    2>&1 | tee "$LOG/ft_skel_${proto}.log"
}

protos=()
if [[ "$PROTO" == "both" ]]; then
  protos=(xsub xacq)
else
  protos=("$PROTO")
fi

for p in "${protos[@]}"; do
  if [[ "$STAGE" == "export" || "$STAGE" == "all" ]]; then
    export_one "$p"
  fi
  if [[ "$STAGE" == "finetune" || "$STAGE" == "all" ]]; then
    finetune_one "$p"
  fi
done

echo "===== DONE ====="
