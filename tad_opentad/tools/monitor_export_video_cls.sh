#!/usr/bin/env bash
# Poll export_video_cls until master.log shows FINISHED; write status file.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR="${1:-$ROOT/logs/export_video_cls_20260707_091713}"
STATUS="$ROOT/logs/export_video_cls_status.txt"
INTERVAL="${INTERVAL:-60}"

echo "monitoring $LOGDIR" > "$STATUS"
while true; do
  if [[ -f "$LOGDIR/master.log" ]] && grep -q 'FINISHED' "$LOGDIR/master.log"; then
    fail=$(grep 'FINISHED' "$LOGDIR/master.log" | tail -1 | grep -oE 'fail=[01]' || echo 'fail=?')
    echo "DONE $(date -Iseconds) $fail" >> "$STATUS"
    grep 'FINISHED\|DONE\|FAIL\|MERGE' "$LOGDIR/master.log" >> "$STATUS"
    json_count=$(ls -1 "$ROOT/data/beiyu/classifiers/"*.json 2>/dev/null | wc -l)
    echo "json_files=$json_count" >> "$STATUS"
    exit 0
  fi
  # progress snapshot
  {
    echo "--- $(date -Iseconds) ---"
    for f in rgb_xsub rgb_xacq skel_xsub_j skel_xsub_b skel_xsub_jm skel_xsub_bm skel_xacq_j skel_xacq_b skel_xacq_jm skel_xacq_bm; do
      log="$LOGDIR/${f}.log"
      if [[ -f "$log" ]] && grep -q '\[ok\]' "$log"; then
        grep '\[ok\]' "$log" | tail -1
      elif [[ -f "$log" ]]; then
        prog=$(grep -oE '[0-9]+/295' "$log" | tail -1)
        echo "$f: ${prog:-starting}"
      fi
    done
    ls -1 "$ROOT/data/beiyu/classifiers/"*.json 2>/dev/null | wc -l | xargs echo "json_files:"
  } >> "$STATUS"
  sleep "$INTERVAL"
done
