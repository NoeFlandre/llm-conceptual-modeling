#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -lt 1 ] || [ "$#" -gt 4 ]; then
  cat >&2 <<'USAGE'
usage: drain_remaining_from_ssh.sh SSH_COMMAND [LOCAL_REPO_DIR] [LOCAL_RESULTS_DIR] [ROOT_NAME_CONTAINS]

Example:
  scripts/vast/drain_remaining_from_ssh.sh \
    "ssh -p 30435 root@59.7.51.43 -L 8080:localhost:8080" \
    /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling \
    /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/results
USAGE
  exit 1
fi

SSH_COMMAND="$1"
LOCAL_REPO_DIR="${2:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
LOCAL_RESULTS_DIR="${3:-$LOCAL_REPO_DIR/results}"
ROOT_NAME_CONTAINS="${4:-${DRAIN_ROOT_NAME_CONTAINS:-}}"
UV_BIN="${UV_BIN:-uv}"
DRAIN_PHASE="${DRAIN_PHASE:-all}"
DRAIN_STATE_FILE="${DRAIN_STATE_FILE:-$LOCAL_RESULTS_DIR/drain-remaining-state.json}"
DRAIN_POLL_SECONDS="${DRAIN_POLL_SECONDS:-30}"
DRAIN_STALE_AFTER_SECONDS="${DRAIN_STALE_AFTER_SECONDS:-3600}"
DRAIN_QUICK_RESUME_SCRIPT="${DRAIN_QUICK_RESUME_SCRIPT:-$SCRIPT_DIR/quick_resume_from_ssh.sh}"

cmd=(
  "$UV_BIN"
  --directory
  "$LOCAL_REPO_DIR"
  run
  lcm
  run
  drain-remaining
  --repo-root
  "$LOCAL_REPO_DIR"
  --results-root
  "$LOCAL_RESULTS_DIR"
  --ssh-command
  "$SSH_COMMAND"
  --state-file
  "$DRAIN_STATE_FILE"
  --phase
  "$DRAIN_PHASE"
  --poll-seconds
  "$DRAIN_POLL_SECONDS"
  --stale-after-seconds
  "$DRAIN_STALE_AFTER_SECONDS"
  --quick-resume-script
  "$DRAIN_QUICK_RESUME_SCRIPT"
)

if [ -n "$ROOT_NAME_CONTAINS" ]; then
  cmd+=(--root-name-contains "$ROOT_NAME_CONTAINS")
fi

if [ "${DRAIN_FULL_COVERAGE:-0}" = "1" ]; then
  cmd+=(--full-coverage)
fi

if [ "${DRAIN_PLAN_ONLY:-0}" = "1" ]; then
  cmd+=(--plan-only)
fi

if [ "${DRAIN_JSON:-0}" = "1" ]; then
  cmd+=(--json)
fi

exec "${cmd[@]}"
