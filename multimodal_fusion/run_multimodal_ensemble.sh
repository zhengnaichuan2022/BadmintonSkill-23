#!/bin/bash
#  Video （）+  ensemble
# : bash run_video_test_scores_2gpu.sh --then-ensemble
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec bash "$ROOT/run_video_test_scores_2gpu.sh" --then-ensemble
