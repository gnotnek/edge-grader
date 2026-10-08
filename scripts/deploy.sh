#!/usr/bin/env bash
# Put the committed repo on the lab boards, the way agentic-edge's
# benchmark/deploy.sh does: rsync from this Mac over the same ssh aliases, so
# the boards need no GitHub access and the repo can stay private.
#
#   scripts/deploy.sh               both boards
#   scripts/deploy.sh pi            one board (pi | jetson)
#   scripts/deploy.sh --allow-dirty include uncommitted changes (recorded)
#
# Per board:
#   1. refuses while a grading run is in progress (never rsync under a run)
#   2. installs uv into ~/.local/bin if it is missing (no sudo)
#   3. rsyncs the tracked files, minus results/ (the boards write their own
#      results/board/, pulled back with scripts/fetch_results.sh)
#   4. uv sync --frozen, then the test suite, on the board
#   5. writes DEPLOYED.json (commit, dirty, time)
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

DIRTY=0
TARGETS=()
for a in "$@"; do
  case "$a" in
    --allow-dirty) DIRTY=1 ;;
    pi|jetson) TARGETS+=("$a") ;;
    all) TARGETS=(pi jetson) ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown argument: $a" >&2; exit 2 ;;
  esac
done
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(pi jetson)

# Same boxes and case convention as agentic-edge (its AGENTS.md §2):
# ~/Research on the Pi, ~/research on the Jetson.
host_of() { case "$1" in pi) echo MITLAB-EDGE ;; jetson) echo MITLAB-JETSON ;; esac; }
repo_of() { case "$1" in pi) echo Research/edge-grader ;; jetson) echo research/edge-grader ;; esac; }
# The Jetson's ~/research is on an exFAT SSD, which has no symlinks, so its venv
# lives on the root filesystem, like agentic-edge's ~/venvs/eval.
venv_env() { case "$1" in jetson) echo 'export UV_PROJECT_ENVIRONMENT=$HOME/venvs/edge-grader;' ;; esac; }

if [ -n "$(git status --porcelain)" ]; then
  if [ $DIRTY -eq 0 ]; then
    echo "uncommitted changes; commit them, or pass --allow-dirty" >&2
    git status --short >&2
    exit 1
  fi
  echo "warning: deploying uncommitted changes (recorded as dirty)"
fi
COMMIT=$(git rev-parse --short HEAD)
LIST=$(mktemp)
trap 'rm -f "$LIST"' EXIT
{ git ls-files
  [ $DIRTY -eq 1 ] && git ls-files --others --exclude-standard
} | grep -v '^results/' | sort -u > "$LIST"

STATUS=0
for b in "${TARGETS[@]}"; do
  H=$(host_of "$b"); R=$(repo_of "$b")
  echo "== $b ($H:~/$R)"

  busy=$(ssh "$H" 'pgrep -fa "[b]oard_grade|[b]oard_queue.py run|[e]dge-grader-(baseline|gsm|news)" | cut -c1-100')
  if [ -n "$busy" ]; then
    echo "   busy, not deploying:" >&2
    echo "$busy" | sed 's/^/     /' >&2
    STATUS=1; continue
  fi

  ssh "$H" '[ -x ~/.local/bin/uv ] || { echo "   installing uv"; curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh >/dev/null; }
            ~/.local/bin/uv --version | sed "s/^/   /"' || { STATUS=1; continue; }

  ssh "$H" "mkdir -p ~/$R" &&
    rsync -a --files-from="$LIST" ./ "$H:$R/" || { echo "   rsync failed" >&2; STATUS=1; continue; }

  if ! ssh "$H" "cd ~/$R && $(venv_env "$b") ~/.local/bin/uv sync --frozen -q &&
                 ~/.local/bin/uv run pytest -q > .deploy-tests.log 2>&1; rc=\$?;
                 tail -1 .deploy-tests.log | sed 's/^/   tests: /'; exit \$rc"; then
    echo "   TESTS FAILED on $b. Log: ~/$R/.deploy-tests.log" >&2
    STATUS=1; continue
  fi

  ssh "$H" "cat > ~/$R/DEPLOYED.json" <<EOF
{"commit": "$COMMIT", "dirty": $([ $DIRTY -eq 1 ] && echo true || echo false), "at": "$(date '+%Y-%m-%dT%H:%M:%S%z')"}
EOF
  echo "   deployed $COMMIT$([ $DIRTY -eq 1 ] && echo ' (dirty)')"
done
exit $STATUS
