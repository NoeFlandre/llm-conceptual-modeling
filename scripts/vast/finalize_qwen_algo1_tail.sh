#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: finalize_qwen_algo1_tail.sh LOCAL_REPO_DIR LOCAL_TAIL_RESULTS_ROOT CANONICAL_RESULTS_ROOT" >&2
  exit 1
fi

LOCAL_REPO_DIR="$1"
LOCAL_TAIL_RESULTS_ROOT="$2"
CANONICAL_RESULTS_ROOT="$3"

mkdir -p "$CANONICAL_RESULTS_ROOT/runs"
rsync -av \
  "$LOCAL_TAIL_RESULTS_ROOT/runs/" \
  "$CANONICAL_RESULTS_ROOT/runs/"

uv --directory "$LOCAL_REPO_DIR" run lcm run refresh-ledger \
  --results-root "$(dirname "$CANONICAL_RESULTS_ROOT")" \
  --ledger-root "$CANONICAL_RESULTS_ROOT" \
  --json
