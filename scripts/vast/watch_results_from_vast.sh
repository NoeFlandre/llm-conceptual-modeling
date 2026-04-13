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
LEDGER_REFRESH_REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LEDGER_RESULTS_DIR="$(cd "$(dirname "$LOCAL_RESULTS_DIR")" && pwd)"
LEDGER_REFRESH_LEDGER_ROOT="$LOCAL_RESULTS_DIR"
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
  LEDGER_PATH_VALUE="$LOCAL_RESULTS_DIR/ledger.json" \
  python3 - <<'PY'
from os import environ
from pathlib import Path
import json

ledger_path = Path(environ["LEDGER_PATH_VALUE"])
ledger_snapshot = None
if ledger_path.exists():
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger_snapshot = {
            "generated_at": ledger.get("generated_at"),
            "expected_total_runs": ledger.get("expected_total_runs"),
            "finished_count": ledger.get("finished_count"),
            "pending_count": ledger.get("pending_count"),
            "retryable_failed_count": ledger.get("retryable_failed_count"),
            "terminal_failed_count": ledger.get("terminal_failed_count"),
        }
    except json.JSONDecodeError:
        ledger_snapshot = None

status_payload = {
    "status": environ["SYNC_STATE"],
    "remote_results_dir": environ["REMOTE_RESULTS_DIR_VALUE"],
    "local_results_dir": environ["LOCAL_RESULTS_DIR_VALUE"],
    "watcher_identity": environ["WATCHER_IDENTITY_VALUE"],
    "updated_at": environ["UPDATED_AT_VALUE"],
    "last_success_at": environ["LAST_SUCCESS_AT_VALUE"],
    "consecutive_failures": int(environ["CONSECUTIVE_FAILURES_VALUE"]),
    "message": environ["SYNC_MESSAGE"],
    "ledger_snapshot": ledger_snapshot,
}

Path(environ["STATUS_PATH"]).write_text(
    json.dumps(status_payload, indent=2, sort_keys=True),
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
  if SSH_PORT="$SSH_PORT" "$FETCH_SCRIPT" "$REMOTE_RESULTS_DIR" "$LOCAL_RESULTS_DIR"; then
    if uv --directory "$LEDGER_REFRESH_REPO_DIR" run lcm run refresh-ledger \
      --results-root "$LEDGER_RESULTS_DIR" \
      --ledger-root "$LEDGER_REFRESH_LEDGER_ROOT"
    then
      LAST_SUCCESS_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf '%s\n' "$LAST_SUCCESS_AT" > "$LOCAL_RESULTS_SYNC_LAST_SUCCESS_PATH"
      CONSECUTIVE_FAILURES=0
      write_sync_status "healthy" "Latest sync succeeded."
      sleep "$SYNC_INTERVAL_SECONDS"
      continue
    fi
  fi
  CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
  write_sync_status "degraded" "Latest sync or ledger refresh failed; watcher will retry."
  sleep "$SYNC_FAILURE_BACKOFF_SECONDS"
done
