#!/usr/bin/env bash
# Keep a copy of everything a board holds besides code: results/ (run folders, pilots and
# interrupted runs in results/local/) and queue/ (job list and per-job logs), in backups/<board>/
# on this Mac, so a board that dies, is wiped or is lent out loses nothing.
# Never deletes anything here; a file the board has changed since the last copy is replaced,
# and the old version kept in backups/<board>/.replaced/<stamp>/.
# backups/ is git-ignored: committed results still come in through fetch_results.sh.
#
#   scripts/backup_board.sh           both boards (an unreachable one is skipped)
#   scripts/backup_board.sh jetson    one board (pi | jetson)
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(pi jetson)
host_of() { case "$1" in pi) echo MITLAB-EDGE ;; jetson) echo MITLAB-JETSON ;; esac; }
repo_of() { case "$1" in pi) echo Research/edge-grader ;; jetson) echo research/edge-grader ;; esac; }

STAMP=$(date +%Y%m%d-%H%M%S)
STATUS=0
for b in "${TARGETS[@]}"; do
  H=$(host_of "$b"); R=$(repo_of "$b")
  [ -n "$H" ] || { echo "unknown board: $b (pi | jetson)" >&2; exit 2; }
  echo "== $b ($H)"
  if ! ssh -o ConnectTimeout=10 "$H" true 2>/dev/null; then
    echo "   not reachable; skipped"; STATUS=1; continue
  fi
  DEST=backups/$b
  for d in results queue; do
    ssh "$H" "[ -d ~/$R/$d ]" || continue
    mkdir -p "$DEST/$d"
    rsync -a --backup --backup-dir="$REPO/$DEST/.replaced/$STAMP/$d" \
      "$H:$R/$d/" "$DEST/$d/" || STATUS=1
  done
  echo "   $(du -sh "$DEST" | cut -f1) in $DEST ($(find "$DEST" -type f -not -path "*/.replaced/*" | wc -l | tr -d ' ') files)"
done
exit $STATUS
