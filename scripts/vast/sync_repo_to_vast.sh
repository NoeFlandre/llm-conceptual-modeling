#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 LOCAL_REPO_DIR USER@HOST:REMOTE_DIR" >&2
  exit 1
fi

LOCAL_REPO_DIR="$1"
REMOTE_TARGET="$2"

rsync -avz \
  --delete \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '.work-venv' \
  --exclude '.ruff_cache' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude 'results' \
  --exclude 'runs' \
  --exclude 'worker-queues' \
  --exclude 'data/results' \
  --exclude 'data/analysis_artifacts' \
  "$LOCAL_REPO_DIR"/ "$REMOTE_TARGET"/
