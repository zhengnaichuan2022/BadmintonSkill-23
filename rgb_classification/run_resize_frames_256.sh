#!/usr/bin/env bash
#  cache/frames (1080p)  cache/frames_256 (short_edge=256)
#
# ：
#   cd ${BADMINTONSKILL23_ROOT}/video
#   bash run_resize_frames_256.sh              #  8 workers
#   bash run_resize_frames_256.sh --workers 16
#   bash run_resize_frames_256.sh --limit 20   # 
#
#  cache：
#   FRAMES_CACHE=frames_256 SKIP_EXTRACT_FRAMES=1 bash run_prepare.sh
#   FRAMES_CACHE=frames_256 bash run_benchmark_4gpu.sh

set -euo pipefail
cd "$(dirname "$0")"

WORKERS="${WORKERS:-8}"
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers) WORKERS="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

echo "Resize frame cache: frames -> frames_256 (short_edge=256)"
echo "Workers: ${WORKERS}"
python3 resize_frames_cache.py --workers "$WORKERS" "${EXTRA[@]}"
