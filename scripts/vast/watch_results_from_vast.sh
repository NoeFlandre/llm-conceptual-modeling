#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 USER@HOST:REMOTE_RESULTS_DIR LOCAL_RESULTS_DIR" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FETCH_SCRIPT="$SCRIPT_DIR/fetch_results_from_vast.sh"
REMOTE_RESULTS_DIR="$1"
LOCAL_RESULTS_DIR="$2"
SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-60}"

if ! [[ "$SYNC_INTERVAL_SECONDS" =~ ^[0-9]+$ ]] || [ "$SYNC_INTERVAL_SECONDS" -le 0 ]; then
  echo "SYNC_INTERVAL_SECONDS must be a positive integer" >&2
  exit 1
fi

while true; do
  "$FETCH_SCRIPT" "$REMOTE_RESULTS_DIR" "$LOCAL_RESULTS_DIR"
  sleep "$SYNC_INTERVAL_SECONDS"
done
