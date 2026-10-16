#!/usr/bin/env bash
# CoM-M2U full pipeline: export logits → fine-tune → ensure each modality improves
set -euo pipefail
ROOT=${BADMINTONSKILL23_ROOT}
MM=$ROOT/multimodal
OUT=$MM/outputs/com_m2u
LOG=$OUT/logs
mkdir -p "$LOG"
export PYTHONPATH=$MM:${PYTHONPATH:-}

TF_PY=/home/ubuntu/users/znc/miniconda3/bin/python
MA_PY=/home/ubuntu/users/syq/miniconda3/envs/RTMLib/bin/python
SK_PY=/home/ubuntu/users/znc/miniconda3/envs/vmamba/bin/python

PROTO=${1:-both}  # xsub|xacq|both
STAGE=${2:-all}   # export|finetune|all

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
  local p_j=$!
  CUDA_VISIBLE_DEVICES=2 $SK_PY -m com_m2u.export_logits --protocol "$proto" --modality skel_streams --stream b --gpu 0 \
    >"$LOG/export_sk_b_${proto}.log" 2>&1 &
  local p_b=$!
  CUDA_VISIBLE_DEVICES=3 $SK_PY -m com_m2u.export_logits --protocol "$proto" --modality skel_streams --stream jm --gpu 0 \
    >"$LOG/export_sk_jm_${proto}.log" 2>&1 &
  local p_jm=$!

  wait $p_rgb $p_j $p_b $p_jm

  CUDA_VISIBLE_DEVICES=1 $SK_PY -m com_m2u.export_logits --protocol "$proto" --modality skel_streams --stream bm --gpu 0 \
    2>&1 | tee "$LOG/export_sk_bm_${proto}.log"
  $TF_PY -m com_m2u.export_logits --protocol "$proto" --modality fuse_skel \
    2>&1 | tee "$LOG/export_sk_fuse_${proto}.log"
}

finetune_one() {
  local proto=$1
  echo "===== FINETUNE $proto ====="
  # IMU (TF) on GPU2, RGB on GPU0, Skeleton on GPU1 — sequential per modality for clarity,
  # but IMU first (fast) then RGB+Skel can be parallel if needed.
  CUDA_VISIBLE_DEVICES=2 $TF_PY -m com_m2u.finetune_imu --protocol "$proto" --auto_search \
    2>&1 | tee "$LOG/ft_imu_${proto}.log"

  CUDA_VISIBLE_DEVICES=0 $MA_PY -m com_m2u.finetune_video --protocol "$proto" --auto_search --device cuda:0 \
    2>&1 | tee "$LOG/ft_rgb_${proto}.log"

  CUDA_VISIBLE_DEVICES=1 $SK_PY -m com_m2u.finetune_skeleton --protocol "$proto" --auto_search --device cuda:0 \
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
$TF_PY - <<'PY'
import json
from pathlib import Path
root=Path("${BADMINTONSKILL23_ROOT}/multimodal/outputs/com_m2u")
rows=[]
for proto in ["xsub","xacq"]:
  for mod in ["imu","rgb","skeleton"]:
    p=root/proto/"results"/f"{mod}_best_summary.json"
    if not p.exists():
      # fallback to com_m2u json
      p=root/proto/"results"/f"{mod}_com_m2u.json"
    if p.exists():
      d=json.loads(p.read_text())
      rows.append((proto,mod,d.get("baseline_acc"),d.get("best_test_acc"),d.get("improved")))
print("protocol modality baseline best improved")
for r in rows:
  print("%s %s %.4f %.4f %s"%r)
PY
