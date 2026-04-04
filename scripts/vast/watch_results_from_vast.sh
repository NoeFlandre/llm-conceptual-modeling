#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "usage: $0 USER@HOST:REMOTE_RESULTS_DIR LOCAL_RESULTS_DIR [SSH_PORT]" >&2
  exit 1
fi

FETCH_SCRIPT="$SCRIPT_DIR/fetch_results_from_vast.sh"
REMOTE_RESULTS_DIR="$1"
LOCAL_RESULTS_DIR="$2"
SSH_PORT="${3:-${SSH_PORT:-22}}"
SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-60}"
SYNC_FAILURE_BACKOFF_SECONDS="${SYNC_FAILURE_BACKOFF_SECONDS:-$SYNC_INTERVAL_SECONDS}"
LOCAL_RESULTS_SYNC_STATUS_PATH="${LOCAL_RESULTS_SYNC_STATUS_PATH:-$LOCAL_RESULTS_DIR/results-sync-status.json}"
LOCAL_RESULTS_SYNC_IDENTITY="${LOCAL_RESULTS_SYNC_IDENTITY:-}"
LOCAL_RESULTS_SYNC_LAST_SUCCESS_PATH="${LOCAL_RESULTS_SYNC_LAST_SUCCESS_PATH:-$LOCAL_RESULTS_DIR/results-sync-last-success.txt}"

vast_require_positive_integer "$SYNC_INTERVAL_SECONDS" "SYNC_INTERVAL_SECONDS"
vast_require_positive_integer "$SYNC_FAILURE_BACKOFF_SECONDS" "SYNC_FAILURE_BACKOFF_SECONDS"

mkdir -p "$LOCAL_RESULTS_DIR"
CONSECUTIVE_FAILURES=0
LAST_SUCCESS_AT=""

write_sync_status() {
  local sync_state="$1"
  local message="$2"
  local timestamp
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  SYNC_STATE="$sync_state" \
  SYNC_MESSAGE="$message" \
  STATUS_PATH="$LOCAL_RESULTS_SYNC_STATUS_PATH" \
  REMOTE_RESULTS_DIR_VALUE="$REMOTE_RESULTS_DIR" \
  LOCAL_RESULTS_DIR_VALUE="$LOCAL_RESULTS_DIR" \
  WATCHER_IDENTITY_VALUE="$LOCAL_RESULTS_SYNC_IDENTITY" \
  UPDATED_AT_VALUE="$timestamp" \
  LAST_SUCCESS_AT_VALUE="$LAST_SUCCESS_AT" \
  CONSECUTIVE_FAILURES_VALUE="$CONSECUTIVE_FAILURES" \
  python3 - <<'PY'
from os import environ
from pathlib import Path
import json

Path(environ["STATUS_PATH"]).write_text(
    json.dumps(
        {
            "status": environ["SYNC_STATE"],
            "remote_results_dir": environ["REMOTE_RESULTS_DIR_VALUE"],
            "local_results_dir": environ["LOCAL_RESULTS_DIR_VALUE"],
            "watcher_identity": environ["WATCHER_IDENTITY_VALUE"],
            "updated_at": environ["UPDATED_AT_VALUE"],
            "last_success_at": environ["LAST_SUCCESS_AT_VALUE"],
            "consecutive_failures": int(environ["CONSECUTIVE_FAILURES_VALUE"]),
            "message": environ["SYNC_MESSAGE"],
        },
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)
PY
}

handle_exit() {
  write_sync_status "stopped" "Watcher stopped."
}

trap handle_exit EXIT

write_sync_status "starting" "Waiting for first sync attempt."

while true; do
  write_sync_status "syncing" "Running result sync."
  if SSH_PORT="$SSH_PORT" "$FETCH_SCRIPT" "$REMOTE_RESULTS_DIR" "$LOCAL_RESULTS_DIR" "$SSH_PORT"; then
    LAST_SUCCESS_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '%s\n' "$LAST_SUCCESS_AT" > "$LOCAL_RESULTS_SYNC_LAST_SUCCESS_PATH"
    CONSECUTIVE_FAILURES=0
    write_sync_status "healthy" "Latest sync succeeded."
    sleep "$SYNC_INTERVAL_SECONDS"
    continue
  fi
  CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
  write_sync_status "degraded" "Latest sync failed; watcher will retry."
  sleep "$SYNC_FAILURE_BACKOFF_SECONDS"
done
