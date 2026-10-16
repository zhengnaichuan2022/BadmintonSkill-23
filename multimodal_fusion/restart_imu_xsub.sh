#!/usr/bin/env bash
set -euo pipefail
ROOT=${BADMINTONSKILL23_ROOT}
OUT=$ROOT/multimodal/outputs/com_m2u
LOG=$OUT/logs
mkdir -p "$LOG"
export PYTHONPATH=$ROOT/multimodal
export TF_XLA_FLAGS=--tf_xla_auto_jit=0
export TF_CPP_MIN_LOG_LEVEL=2
export HDF5_USE_FILE_LOCKING=FALSE
# Force CPU: model is tiny; GPUs are contended/OOM
export CUDA_VISIBLE_DEVICES=

echo "[$(date)] start IMU FT xsub (CPU)" | tee -a "$LOG/restart.log"
/home/ubuntu/users/znc/miniconda3/bin/python -m com_m2u.finetune_imu --protocol xsub --auto_search \
  2>&1 | tee "$LOG/ft_imu_xsub.log"
echo "[$(date)] IMU FT finished exit=$?" | tee -a "$LOG/restart.log"
