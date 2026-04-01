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

vast_require_positive_integer "$SYNC_INTERVAL_SECONDS" "SYNC_INTERVAL_SECONDS"

while true; do
  "$FETCH_SCRIPT" "$REMOTE_RESULTS_DIR" "$LOCAL_RESULTS_DIR"
  sleep "$SYNC_INTERVAL_SECONDS"
done
