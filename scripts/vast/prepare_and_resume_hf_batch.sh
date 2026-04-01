#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

if [ "$#" -lt 6 ] || [ "$#" -gt 7 ]; then
  cat >&2 <<'USAGE'
usage: prepare_and_resume_hf_batch.sh SSH_TARGET SSH_PORT LOCAL_REPO_DIR REMOTE_REPO_DIR CONFIG_RELATIVE_PATH REMOTE_RESULTS_DIR [LOCAL_RESULTS_DIR]

Example:
  scripts/vast/prepare_and_resume_hf_batch.sh \
    root@61.228.57.170 \
    31291 \
    /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling \
    /workspace/llm-conceptual-modeling \
    configs/hf_transformers_paper_batch.yaml \
    /workspace/results/hf-paper-batch \
    /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/results/hf-paper-batch

Optional smoke gate environment variables:
  SMOKE_ALGORITHM
  SMOKE_MODEL
  SMOKE_PAIR_NAME
  SMOKE_CONDITION_BITS
  SMOKE_DECODING
  SMOKE_REPLICATION
  SMOKE_OUTPUT_ROOT
USAGE
  exit 1
fi

SSH_TARGET="$1"
SSH_PORT="$2"
LOCAL_REPO_DIR="$3"
REMOTE_REPO_DIR="$4"
CONFIG_RELATIVE_PATH="$5"
REMOTE_RESULTS_DIR="$6"
LOCAL_RESULTS_DIR="${7:-}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_rsa}"
REMOTE_CONFIG_PATH="$REMOTE_REPO_DIR/$CONFIG_RELATIVE_PATH"
REMOTE_EFFECTIVE_CONFIG_PATH="${REMOTE_EFFECTIVE_CONFIG_PATH:-$REMOTE_RESULTS_DIR/runtime_config.yaml}"
REMOTE_PREVIEW_DIR="${REMOTE_PREVIEW_DIR:-$REMOTE_RESULTS_DIR/preview}"
REMOTE_RUN_LOG="${REMOTE_RUN_LOG:-$REMOTE_RESULTS_DIR/run.log}"
REMOTE_PID_PATH="${REMOTE_PID_PATH:-$REMOTE_RESULTS_DIR/batch.pid}"
SMOKE_REPLICATION="${SMOKE_REPLICATION:-0}"
SMOKE_OUTPUT_ROOT="${SMOKE_OUTPUT_ROOT:-$REMOTE_RESULTS_DIR/smoke}"
BATCH_GENERATION_TIMEOUT_SECONDS="${BATCH_GENERATION_TIMEOUT_SECONDS:-}"
BATCH_RESUME_PASS_MODE="${BATCH_RESUME_PASS_MODE:-}"
BATCH_RETRY_TIMEOUT_FAILURES_ON_RESUME="${BATCH_RETRY_TIMEOUT_FAILURES_ON_RESUME:-}"
BATCH_WORKER_PROCESS_MODE="${BATCH_WORKER_PROCESS_MODE:-}"
BATCH_MAX_REQUESTS_PER_WORKER_PROCESS="${BATCH_MAX_REQUESTS_PER_WORKER_PROCESS:-}"
LOCAL_RESULTS_SYNC_INTERVAL_SECONDS="${LOCAL_RESULTS_SYNC_INTERVAL_SECONDS:-}"
LOCAL_RESULTS_SYNC_LOG_PATH="${LOCAL_RESULTS_SYNC_LOG_PATH:-}"
LOCAL_RESULTS_SYNC_PID_PATH="${LOCAL_RESULTS_SYNC_PID_PATH:-}"

SSH_CMD=($(vast_ssh_command "$SSH_PORT" "$SSH_KEY_PATH"))
RSYNC_SSH="$(vast_rsync_ssh_command "$SSH_PORT" "$SSH_KEY_PATH")"

echo "[1/6] Sync repository"
rsync -avz \
  -e "$RSYNC_SSH" \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude 'data/results' \
  --exclude 'data/analysis_artifacts' \
  "$LOCAL_REPO_DIR"/ "$SSH_TARGET:$REMOTE_REPO_DIR"/

if vast_has_value "$LOCAL_RESULTS_DIR"; then
  echo "[2/6] Seed remote results root from local copy"
  mkdir -p "$LOCAL_RESULTS_DIR"
  rsync -avz -e "$RSYNC_SSH" "$LOCAL_RESULTS_DIR"/ "$SSH_TARGET:$REMOTE_RESULTS_DIR"/
else
  echo "[2/6] No local results seed provided; skipping"
fi

if vast_has_value "$LOCAL_RESULTS_DIR" && vast_has_value "$LOCAL_RESULTS_SYNC_INTERVAL_SECONDS"; then
  echo "[sync] Launch local result autosync"
  if [ -z "$LOCAL_RESULTS_SYNC_LOG_PATH" ]; then
    LOCAL_RESULTS_SYNC_LOG_PATH="$LOCAL_RESULTS_DIR/results-sync.log"
  fi
  if [ -z "$LOCAL_RESULTS_SYNC_PID_PATH" ]; then
    LOCAL_RESULTS_SYNC_PID_PATH="$LOCAL_RESULTS_DIR/results-sync.pid"
  fi
  mkdir -p "$LOCAL_RESULTS_DIR"
  pkill -f "scripts/vast/watch_results_from_vast.sh $SSH_TARGET:$REMOTE_RESULTS_DIR $LOCAL_RESULTS_DIR" || true
  SSH_KEY_PATH="$SSH_KEY_PATH" \
  SSH_PORT="$SSH_PORT" \
  SYNC_INTERVAL_SECONDS="$LOCAL_RESULTS_SYNC_INTERVAL_SECONDS" \
  nohup bash "$LOCAL_REPO_DIR/scripts/vast/watch_results_from_vast.sh" \
    "$SSH_TARGET:$REMOTE_RESULTS_DIR" \
    "$LOCAL_RESULTS_DIR" \
    >"$LOCAL_RESULTS_SYNC_LOG_PATH" 2>&1 </dev/null &
  echo $! > "$LOCAL_RESULTS_SYNC_PID_PATH"
fi

echo "[3/6] Bootstrap remote runtime"
"${SSH_CMD[@]}" "$SSH_TARGET" "bash '$REMOTE_REPO_DIR/scripts/vast/bootstrap_gpu_host.sh' '$REMOTE_REPO_DIR'"

echo "[4/6] Run doctor + config preview"
"${SSH_CMD[@]}" "$SSH_TARGET" "
  set -euo pipefail
  cd '$REMOTE_REPO_DIR'
  mkdir -p '$REMOTE_RESULTS_DIR'
  export BATCH_GENERATION_TIMEOUT_SECONDS='$BATCH_GENERATION_TIMEOUT_SECONDS'
  export BATCH_RESUME_PASS_MODE='$BATCH_RESUME_PASS_MODE'
  export BATCH_RETRY_TIMEOUT_FAILURES_ON_RESUME='$BATCH_RETRY_TIMEOUT_FAILURES_ON_RESUME'
  export BATCH_WORKER_PROCESS_MODE='$BATCH_WORKER_PROCESS_MODE'
  export BATCH_MAX_REQUESTS_PER_WORKER_PROCESS='$BATCH_MAX_REQUESTS_PER_WORKER_PROCESS'
  python3 - <<'PY'
from pathlib import Path
import os
import yaml

source_path = Path('$REMOTE_CONFIG_PATH')
target_path = Path('$REMOTE_EFFECTIVE_CONFIG_PATH')
payload = yaml.safe_load(source_path.read_text(encoding='utf-8'))
context_policy = dict(payload['runtime'].get('context_policy', {}))

timeout_value = os.environ.get('BATCH_GENERATION_TIMEOUT_SECONDS', '').strip()
if timeout_value:
    context_policy['generation_timeout_seconds'] = float(timeout_value)

resume_pass_mode = os.environ.get('BATCH_RESUME_PASS_MODE', '').strip()
if resume_pass_mode:
    context_policy['resume_pass_mode'] = resume_pass_mode

retry_timeout = os.environ.get('BATCH_RETRY_TIMEOUT_FAILURES_ON_RESUME', '').strip().lower()
if retry_timeout:
    context_policy['retry_timeout_failures_on_resume'] = retry_timeout in {'1', 'true', 'yes', 'on'}

worker_process_mode = os.environ.get('BATCH_WORKER_PROCESS_MODE', '').strip()
if worker_process_mode:
    context_policy['worker_process_mode'] = worker_process_mode

max_requests = os.environ.get('BATCH_MAX_REQUESTS_PER_WORKER_PROCESS', '').strip()
if max_requests:
    context_policy['max_requests_per_worker_process'] = int(float(max_requests))

payload['runtime']['context_policy'] = context_policy
target_path.parent.mkdir(parents=True, exist_ok=True)
target_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
PY
  .venv/bin/lcm doctor --json --results-root '$REMOTE_RESULTS_DIR'
  .venv/bin/lcm run validate-config --config '$REMOTE_EFFECTIVE_CONFIG_PATH' --output-dir '$REMOTE_PREVIEW_DIR'
"

if vast_has_value "${SMOKE_ALGORITHM:-}" \
  && vast_has_value "${SMOKE_MODEL:-}" \
  && vast_has_value "${SMOKE_PAIR_NAME:-}" \
  && vast_has_value "${SMOKE_CONDITION_BITS:-}" \
  && vast_has_value "${SMOKE_DECODING:-}"; then
  echo "[5/6] Run smoke gate"
  "${SSH_CMD[@]}" "$SSH_TARGET" "
    set -euo pipefail
    cd '$REMOTE_REPO_DIR'
    .venv/bin/lcm run smoke \
      --config '$REMOTE_EFFECTIVE_CONFIG_PATH' \
      --algorithm '${SMOKE_ALGORITHM}' \
      --model '${SMOKE_MODEL}' \
      --pair-name '${SMOKE_PAIR_NAME}' \
      --condition-bits '${SMOKE_CONDITION_BITS}' \
      --decoding '${SMOKE_DECODING}' \
      --replication '${SMOKE_REPLICATION}' \
      --output-root '${SMOKE_OUTPUT_ROOT}'
  "
else
  echo "[5/6] Smoke gate not configured; skipping"
fi

echo "[6/6] Launch resumable batch"
"${SSH_CMD[@]}" "$SSH_TARGET" "
  set -euo pipefail
  cd '$REMOTE_REPO_DIR'
  mkdir -p '$REMOTE_RESULTS_DIR'
  pkill -f '.venv/bin/lcm run paper-batch --config $REMOTE_EFFECTIVE_CONFIG_PATH --resume' || true
  pkill -f 'llm_conceptual_modeling.hf_worker' || true
  sleep 2
  nohup .venv/bin/lcm run paper-batch --config '$REMOTE_EFFECTIVE_CONFIG_PATH' --resume >'$REMOTE_RUN_LOG' 2>&1 </dev/null &
  sleep 2
  pgrep -n -f '.venv/bin/lcm run paper-batch --config $REMOTE_EFFECTIVE_CONFIG_PATH --resume' > '$REMOTE_PID_PATH'
  cat '$REMOTE_PID_PATH'
"
