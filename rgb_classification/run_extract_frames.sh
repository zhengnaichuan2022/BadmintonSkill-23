#!/usr/bin/env bash
# （）； config  Resize/Crop。
#
# ：
#   cd ${BADMINTONSKILL23_ROOT}/video
#   bash run_extract_frames.sh
#   bash run_extract_frames.sh --workers 8

set -euo pipefail
cd "$(dirname "$0")"

python3 prepare_annotations.py
python3 extract_frames.py "$@"

echo "Frames ready under: $(pwd)/cache/frames/"
