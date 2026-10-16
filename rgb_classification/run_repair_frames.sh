#!/usr/bin/env bash
#  cache 「」 instance，。
#
# ：
#   cd ${BADMINTONSKILL23_ROOT}/video
#   bash run_repair_frames.sh --list          # ，
#   bash run_repair_frames.sh                 # 
#   bash run_repair_frames.sh --workers 8

set -euo pipefail
cd "$(dirname "$0")"

WORKERS=4
LIST_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers) WORKERS="$2"; shift 2 ;;
    --list) LIST_ONLY=1; shift ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

if [[ "$LIST_ONLY" == "1" ]]; then
  python3 extract_frames.py --list-wrong-video
  exit 0
fi

echo "Scanning wrong-video cache entries..."
python3 extract_frames.py --list-wrong-video

echo ""
echo "Re-extracting wrong instances (${WORKERS} workers)..."
python3 extract_frames.py --repair-wrong-video --workers "$WORKERS"

echo ""
echo "Refreshing frame annotations from cache..."
python3 prepare_annotations.py

echo "Done. Check: cache/frames/repair_manifest.jsonl"
