#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

if [ "$#" -ne 2 ]; then
  echo "usage: $0 USER@HOST:REMOTE_RESULTS_DIR LOCAL_RESULTS_DIR" >&2
  exit 1
fi

FETCH_SCRIPT="$SCRIPT_DIR/fetch_results_from_vast.sh"
REMOTE_RESULTS_DIR="$1"
LOCAL_RESULTS_DIR="$2"
SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-60}"
SYNC_FAILURE_BACKOFF_SECONDS="${SYNC_FAILURE_BACKOFF_SECONDS:-$SYNC_INTERVAL_SECONDS}"
LOCAL_RESULTS_SYNC_STATUS_PATH="${LOCAL_RESULTS_SYNC_STATUS_PATH:-$LOCAL_RESULTS_DIR/results-sync-status.json}"
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
  python3 - <<PY
from pathlib import Path
import json

Path(${LOCAL_RESULTS_SYNC_STATUS_PATH@Q}).write_text(
    json.dumps(
        {
            "status": ${sync_state@Q},
            "remote_results_dir": ${REMOTE_RESULTS_DIR@Q},
            "local_results_dir": ${LOCAL_RESULTS_DIR@Q},
            "updated_at": ${timestamp@Q},
            "last_success_at": ${LAST_SUCCESS_AT@Q},
            "consecutive_failures": ${CONSECUTIVE_FAILURES},
            "message": ${message@Q},
        },
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)
PY
}

write_sync_status "starting" "Waiting for first sync attempt."

while true; do
  if "$FETCH_SCRIPT" "$REMOTE_RESULTS_DIR" "$LOCAL_RESULTS_DIR"; then
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
