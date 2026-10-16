#!/usr/bin/env bash
# ：/、、
#
# ：
#   cd ${BADMINTONSKILL23_ROOT}/video
#   bash show_cache_progress.sh              # 
#   bash show_cache_progress.sh --watch 5    #  5 

set -euo pipefail
cd "$(dirname "$0")"
python3 show_cache_progress.py "$@"
