#!/usr/bin/env bash
# （）
#
# ：
#   cd ${BADMINTONSKILL23_ROOT}/video
#   bash show_progress.sh

set -euo pipefail
cd "$(dirname "$0")"
python3 show_progress.py "$@"
