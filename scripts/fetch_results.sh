#!/usr/bin/env bash
# Pull the boards' run folders (results/board/) back into this repo, to commit.
# Never deletes or overwrites anything already here.
#
#   scripts/fetch_results.sh          both boards
#   scripts/fetch_results.sh pi       one board (pi | jetson)
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(pi jetson)
host_of() { case "$1" in pi) echo MITLAB-EDGE ;; jetson) echo MITLAB-JETSON ;; esac; }
repo_of() { case "$1" in pi) echo Research/edge-grader ;; jetson) echo research/edge-grader ;; esac; }

mkdir -p results/board
STATUS=0
for b in "${TARGETS[@]}"; do
  H=$(host_of "$b"); R=$(repo_of "$b")
  [ -n "$H" ] || { echo "unknown board: $b (pi | jetson)" >&2; exit 2; }
  echo "== $b ($H)"
  ssh "$H" "[ -d ~/$R/results/board ]" || { echo "   no results/board yet"; continue; }
  rsync -a --ignore-existing --itemize-changes "$H:$R/results/board/" results/board/ |
    awk '$1 ~ /^>f/ {print "   " $2}' || STATUS=1
done
echo "then: git add results/board && git commit"
exit $STATUS
