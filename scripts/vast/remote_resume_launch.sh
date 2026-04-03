#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

if [ "$#" -ne 5 ]; then
  cat >&2 <<'USAGE'
usage: remote_resume_launch.sh REMOTE_REPO_DIR REMOTE_RESULTS_DIR REMOTE_EFFECTIVE_CONFIG_PATH REMOTE_RUN_LOG REMOTE_PID_PATH
USAGE
  exit 1
fi

REMOTE_REPO_DIR="$1"
REMOTE_RESULTS_DIR="$2"
REMOTE_EFFECTIVE_CONFIG_PATH="$3"
REMOTE_RUN_LOG="$4"
REMOTE_PID_PATH="$5"

cd "$REMOTE_REPO_DIR"
mkdir -p "$REMOTE_RESULTS_DIR"

pkill -f 'llm_conceptual_modeling.hf_worker' || true
sleep 2
nohup .venv/bin/lcm run paper-batch --config "$REMOTE_EFFECTIVE_CONFIG_PATH" --resume >"$REMOTE_RUN_LOG" 2>&1 </dev/null &
sleep 2
pgrep -n -f ".venv/bin/lcm run paper-batch --config $REMOTE_EFFECTIVE_CONFIG_PATH --resume" > "$REMOTE_PID_PATH"
cat "$REMOTE_PID_PATH"
