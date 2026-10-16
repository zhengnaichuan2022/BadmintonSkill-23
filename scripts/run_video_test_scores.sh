#!/usr/bin/env bash
# Video test： test_scores.pkl（r2plus1d ·  Kinetics）
#
# :
#   bash run_video_test_scores.sh xsub          #  GPU 0
#   bash run_video_test_scores.sh xacq 1       #  GPU 1
#   CUDA_VISIBLE_DEVICES=2 bash run_video_test_scores.sh xsub
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PROTOCOL="${1:?protocol required: xsub|xacq}"
GPU="${2:-${CUDA_VISIBLE_DEVICES:-0}}"

export MMACTION_PYTHON="${MMACTION_PYTHON:-/home/ubuntu/users/syq/miniconda3/envs/RTMLib/bin/python}"
export MMACTION2_ROOT="${MMACTION2_ROOT:-/home/ubuntu/users/dyc/code/mmaction2}"
export PYTHONPATH="${MMACTION2_ROOT}:${ROOT}/video:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

LOG_DIR="${LOG_DIR:-$ROOT/multimodal/logs}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/export_video_${PROTOCOL}_gpu${GPU}_$(date +%Y%m%d_%H%M%S).log"

echo "=== Video test scores: $PROTOCOL on GPU $GPU ===" | tee "$LOG"
echo "MMACTION_PYTHON=$MMACTION_PYTHON" | tee -a "$LOG"
echo "MMACTION2_ROOT=$MMACTION2_ROOT" | tee -a "$LOG"
echo "Started: $(date -Iseconds)" | tee -a "$LOG"

cd "$ROOT"
CUDA_VISIBLE_DEVICES="$GPU" python3 export_test_scores.py --protocol "$PROTOCOL" --modality video 2>&1 | tee -a "$LOG"

python3 - <<PYEOF | tee -a "$LOG"
import json, pickle
from pathlib import Path
ROOT = Path("$ROOT")
proto = "$PROTOCOL"
pkl = ROOT / f"video/outputs/{proto}/r2plus1d_badminton_r34_kinetics/test_scores.pkl"
ref = ROOT / f"video/outputs/{proto}/r2plus1d_badminton_r34_kinetics/test_results.json"
d = pickle.load(open(pkl, "rb"))
acc = float((d["scores"].argmax(1) == d["labels"]).mean())
print(f"VERIFY {proto}: test_scores.pkl acc={acc:.4f}")
if ref.is_file():
    r = json.loads(ref.read_text())["accuracy"]
    print(f"VERIFY {proto}: test_results.json ref={r:.4f} delta={acc-r:+.4f}")
PYEOF

echo "Finished: $(date -Iseconds)" | tee -a "$LOG"
echo "Log: $LOG"
