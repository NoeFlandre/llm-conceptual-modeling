#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 USER@HOST:REMOTE_RESULTS_DIR LOCAL_RESULTS_DIR" >&2
  exit 1
fi

REMOTE_RESULTS_DIR="$1"
LOCAL_RESULTS_DIR="$2"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_rsa}"
SSH_PORT="${SSH_PORT:-22}"
RSYNC_SSH="ssh -i $SSH_KEY_PATH -p $SSH_PORT"

mkdir -p "$LOCAL_RESULTS_DIR"
rsync -avz -e "$RSYNC_SSH" "$REMOTE_RESULTS_DIR"/ "$LOCAL_RESULTS_DIR"/
