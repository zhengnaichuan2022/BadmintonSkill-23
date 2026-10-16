#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/ubuntu/users/syq/miniconda3/envs/RTMLib/bin/python"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

export PYTHONPATH="/home/ubuntu/users/syq/mmpose"
#  Python  CUDA_VISIBLE_DEVICES， GPU
unset CUDA_VISIBLE_DEVICES

GPUS="${GPUS:-0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/batch_pose_extract_${STAMP}.log"

echo "Log: ${LOG_FILE}"
echo "Start: $(date)"

cd "${SCRIPT_DIR}"

ARGS=("$@")
if [ "${#ARGS[@]}" -gt 0 ] && [ "${ARGS[0]}" = "--" ]; then
  ARGS=("${ARGS[@]:1}")
fi

has_gpus_arg=false
for i in "${!ARGS[@]}"; do
  if [ "${ARGS[$i]}" = "--gpus" ]; then
    has_gpus_arg=true
    GPUS="${ARGS[$((i + 1))]}"
    break
  elif [[ "${ARGS[$i]}" == --gpus=* ]]; then
    has_gpus_arg=true
    GPUS="${ARGS[$i]#--gpus=}"
    break
  fi
done

echo "GPUS: ${GPUS}"

EXTRA_ARGS=()
if ! printf '%s\n' "${ARGS[@]}" | grep -qE '^--shard-id$|^--shard-id='; then
  if [ "${has_gpus_arg}" = true ] || [[ "${GPUS}" == *,* ]]; then
    if [ "${has_gpus_arg}" = false ]; then
      EXTRA_ARGS=(--gpus "${GPUS}")
    fi
  else
    export CUDA_VISIBLE_DEVICES="${GPUS}"
    EXTRA_ARGS=(--device "cuda:0")
  fi
fi

"${PYTHON}" batch_pose_extract_r2.py "${EXTRA_ARGS[@]}" "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"

echo "Done: $(date)"
