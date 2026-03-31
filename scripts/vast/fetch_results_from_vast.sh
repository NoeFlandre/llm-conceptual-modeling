#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 USER@HOST:REMOTE_RESULTS_DIR LOCAL_RESULTS_DIR" >&2
  exit 1
fi

REMOTE_RESULTS_DIR="$1"
LOCAL_RESULTS_DIR="$2"

mkdir -p "$LOCAL_RESULTS_DIR"
rsync -avz "$REMOTE_RESULTS_DIR"/ "$LOCAL_RESULTS_DIR"/
