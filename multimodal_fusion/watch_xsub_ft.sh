#!/usr/bin/env bash
set -euo pipefail
ROOT=${BADMINTONSKILL23_ROOT}
OUT=$ROOT/multimodal/outputs/com_m2u
LOG=$OUT/logs
mkdir -p "$LOG"
export PYTHONPATH=$ROOT/multimodal:$ROOT/skeleton/pyskl-main/pyskl-main:$ROOT/video:/home/ubuntu/users/dyc/code/mmaction2:${PYTHONPATH:-}
TF=/home/ubuntu/users/znc/miniconda3/bin/python
MA=/home/ubuntu/users/syq/miniconda3/envs/RTMLib/bin/python
SK=/home/ubuntu/users/znc/miniconda3/envs/vmamba/bin/python

echo "[watch] start $(date)" | tee -a "$LOG/watch.log"

while [ ! -f "$OUT/xsub/logits/skel_bm/train.npz" ]; do
  echo "[watch] waiting skel_bm $(date)" | tee -a "$LOG/watch.log"
  sleep 20
done
echo "[watch] xsub bm done" | tee -a "$LOG/watch.log"

while [ ! -f "$OUT/xsub/logits/rgb/train.npz" ]; do
  echo "[watch] waiting rgb train $(date)" | tee -a "$LOG/watch.log"
  sleep 30
done
echo "[watch] xsub rgb train done" | tee -a "$LOG/watch.log"

$TF -m com_m2u.export_logits --protocol xsub --modality fuse_skel 2>&1 | tee -a "$LOG/fuse_xsub.log"
echo "[watch] fused; start IMU FT xsub" | tee -a "$LOG/watch.log"
CUDA_VISIBLE_DEVICES=2 $TF -m com_m2u.finetune_imu --protocol xsub --auto_search 2>&1 | tee "$LOG/ft_imu_xsub.log"
echo "[watch] IMU xsub done $(date)" | tee -a "$LOG/watch.log"
