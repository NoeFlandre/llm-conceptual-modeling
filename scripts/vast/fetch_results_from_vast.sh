#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "usage: $0 USER@HOST:REMOTE_RESULTS_DIR LOCAL_RESULTS_DIR [SSH_PORT]" >&2
  exit 1
fi

REMOTE_RESULTS_DIR="$1"
LOCAL_RESULTS_DIR="$2"
SSH_PORT="${3:-${SSH_PORT:-22}}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_rsa}"
RSYNC_TIMEOUT_SECONDS="${RSYNC_TIMEOUT_SECONDS:-60}"
LOCAL_RESULTS_SYNC_EXCLUDES="${LOCAL_RESULTS_SYNC_EXCLUDES:-ledger.json results-sync-status.json results-sync-last-success.txt}"
RSYNC_SSH="$(vast_rsync_ssh_command "$SSH_PORT" "$SSH_KEY_PATH")"
RSYNC_RESUME_FLAGS="$(vast_rsync_resume_flags "$RSYNC_TIMEOUT_SECONDS")"

RSYNC_EXCLUDE_FLAGS=()
if [ -n "$LOCAL_RESULTS_SYNC_EXCLUDES" ]; then
  for pattern in $LOCAL_RESULTS_SYNC_EXCLUDES; do
    RSYNC_EXCLUDE_FLAGS+=(--exclude "$pattern")
  done
fi

vast_require_positive_integer "$RSYNC_TIMEOUT_SECONDS" "RSYNC_TIMEOUT_SECONDS"

mkdir -p "$LOCAL_RESULTS_DIR"
set +e
if [ "${#RSYNC_EXCLUDE_FLAGS[@]}" -gt 0 ]; then
  rsync -avz \
    $RSYNC_RESUME_FLAGS \
    "${RSYNC_EXCLUDE_FLAGS[@]}" \
    -e "$RSYNC_SSH" \
    "$REMOTE_RESULTS_DIR"/ "$LOCAL_RESULTS_DIR"/
else
  rsync -avz \
    $RSYNC_RESUME_FLAGS \
    -e "$RSYNC_SSH" \
    "$REMOTE_RESULTS_DIR"/ "$LOCAL_RESULTS_DIR"/
fi
rsync_status=$?
set -e

if [ "$rsync_status" -eq 24 ]; then
  echo "rsync reported file vanished during sync; treating as non-fatal for active results trees." >&2
  exit 0
fi

exit "$rsync_status"
