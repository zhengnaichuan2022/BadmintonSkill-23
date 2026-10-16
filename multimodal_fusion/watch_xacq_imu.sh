#!/usr/bin/env bash
set -euo pipefail
ROOT=${BADMINTONSKILL23_ROOT}
OUT=$ROOT/multimodal/outputs/com_m2u
LOG=$OUT/logs
export PYTHONPATH=$ROOT/multimodal
export HDF5_USE_FILE_LOCKING=FALSE
export TF_CPP_MIN_LOG_LEVEL=2

echo "[$(date)] wait xacq rgb train.npz"
while [ ! -f "$OUT/xacq/logits/rgb/train.npz" ]; do sleep 30; done
echo "[$(date)] xacq rgb ready; start IMU FT"
/home/ubuntu/users/znc/miniconda3/bin/python -m com_m2u.finetune_imu_simple \
  --protocol xacq --mode com_m2u --epochs 12 --lambda_kd 0.3 --gamma 2.0 --lr 3e-5 --gpu 2 --bs 64 \
  2>&1 | tee "$LOG/ft_imu_xacq.log"
echo "[$(date)] IMU xacq done"
