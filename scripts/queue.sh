#!/usr/bin/env bash
# Drive a board's edge-grader job queue (scripts/board_queue.py) from the Mac, over the same
# ssh aliases as deploy.sh.
#
#   scripts/queue.sh jetson add --preset phase0
#   scripts/queue.sh pi add BENCH=gsm INSTANCES=1 -- e2b symbolic
#   scripts/queue.sh jetson start | status | stop | cancel 3 | retry 3
#   scripts/queue.sh pi log            # the worker's log; log 3 = job 3's log
set -uo pipefail
case "${1:-}" in
  pi) H=MITLAB-EDGE; R=Research/edge-grader ;;
  jetson) H=MITLAB-JETSON; R=research/edge-grader ;;
  *) sed -n '2,9p' "$0"; exit 2 ;;
esac
shift
[ $# -gt 0 ] || { echo "usage: $0 pi|jetson <add|status|start|stop|cancel|retry|log> ..." >&2; exit 2; }
if [ "$1" = log ]; then
  f=queue/worker.log; [ -n "${2:-}" ] && f=queue/logs/job-$2.log
  exec ssh "$H" "tail -n 40 ~/$R/$f"
fi
exec ssh "$H" "cd ~/$R && python3 scripts/board_queue.py $(printf '%q ' "$@")"
