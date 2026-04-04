#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

if [ "$#" -ne 4 ]; then
  cat >&2 <<'USAGE'
usage: quick_resume_from_ssh.sh SSH_COMMAND CONFIG_PATH REMOTE_RESULTS_DIR LOCAL_RESULTS_DIR

Example:
  scripts/vast/quick_resume_from_ssh.sh \
    "ssh -p 35895 root@1.193.139.231 -L 8080:localhost:8080" \
    results/hf-paper-batch-algo1-olmo-current/runtime_config.yaml \
    /workspace/results/hf-paper-batch-algo1-olmo-current \
    /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/results/hf-paper-batch-algo1-olmo-current
USAGE
  exit 1
fi

SSH_COMMAND="$1"
CONFIG_PATH="$2"
REMOTE_RESULTS_DIR="$3"
LOCAL_RESULTS_DIR="$4"
LOCAL_REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-/workspace/llm-conceptual-modeling}"

PARSED_SSH="$(vast_parse_ssh_command "$SSH_COMMAND")"

SSH_TARGET="$(printf '%s\n' "$PARSED_SSH" | sed -n '1p')"
SSH_PORT="$(printf '%s\n' "$PARSED_SSH" | sed -n '2p')"

exec "$SCRIPT_DIR/prepare_and_resume_hf_batch.sh" \
  "$SSH_TARGET" \
  "$SSH_PORT" \
  "$LOCAL_REPO_DIR" \
  "$REMOTE_REPO_DIR" \
  "$CONFIG_PATH" \
  "$REMOTE_RESULTS_DIR" \
  "$LOCAL_RESULTS_DIR"
