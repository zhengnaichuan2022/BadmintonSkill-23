#!/usr/bin/env bash
#  benchmark （ nohup ）
set +e
pkill -f "run_benchmark_scratch.sh"
pkill -f "run_benchmark.sh"
pkill -f "${BADMINTONSKILL23_ROOT}/video/configs/mmaction"
pkill -f "run_train.sh xsub"
pkill -f "run_train.sh xacq"
sleep 1
if pgrep -f "${BADMINTONSKILL23_ROOT}/video" >/dev/null 2>&1; then
  echo "，..."
  pkill -9 -f "${BADMINTONSKILL23_ROOT}/video"
fi
echo ""
