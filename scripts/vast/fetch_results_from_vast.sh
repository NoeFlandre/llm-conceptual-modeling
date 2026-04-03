#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

if [ "$#" -ne 2 ]; then
  echo "usage: $0 USER@HOST:REMOTE_RESULTS_DIR LOCAL_RESULTS_DIR" >&2
  exit 1
fi

REMOTE_RESULTS_DIR="$1"
LOCAL_RESULTS_DIR="$2"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_rsa}"
SSH_PORT="${SSH_PORT:-22}"
RSYNC_TIMEOUT_SECONDS="${RSYNC_TIMEOUT_SECONDS:-60}"
RSYNC_SSH="$(vast_rsync_ssh_command "$SSH_PORT" "$SSH_KEY_PATH")"

vast_require_positive_integer "$RSYNC_TIMEOUT_SECONDS" "RSYNC_TIMEOUT_SECONDS"

mkdir -p "$LOCAL_RESULTS_DIR"
rsync -avz \
  --partial --append-verify --timeout "$RSYNC_TIMEOUT_SECONDS" \
  -e "$RSYNC_SSH" \
  "$REMOTE_RESULTS_DIR"/ "$LOCAL_RESULTS_DIR"/
