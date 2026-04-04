#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

if [ "$#" -lt 6 ] || [ "$#" -gt 7 ]; then
  cat >&2 <<'USAGE'
usage: prepare_and_resume_hf_batch.sh SSH_TARGET SSH_PORT LOCAL_REPO_DIR REMOTE_REPO_DIR CONFIG_PATH REMOTE_RESULTS_DIR [LOCAL_RESULTS_DIR]

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
CONFIG_PATH="$5"
REMOTE_RESULTS_DIR="$6"
LOCAL_RESULTS_DIR="${7:-}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_rsa}"
REMOTE_EFFECTIVE_CONFIG_PATH="${REMOTE_EFFECTIVE_CONFIG_PATH:-$REMOTE_RESULTS_DIR/runtime_config.yaml}"
REMOTE_PREVIEW_DIR="${REMOTE_PREVIEW_DIR:-$REMOTE_RESULTS_DIR/preview}"
REMOTE_RUN_LOG="${REMOTE_RUN_LOG:-$REMOTE_RESULTS_DIR/run.log}"
REMOTE_PID_PATH="${REMOTE_PID_PATH:-$REMOTE_RESULTS_DIR/batch.pid}"
SMOKE_REPLICATION="${SMOKE_REPLICATION:-0}"
SMOKE_OUTPUT_ROOT="${SMOKE_OUTPUT_ROOT:-$REMOTE_RESULTS_DIR/smoke}"
BATCH_GENERATION_TIMEOUT_SECONDS="${BATCH_GENERATION_TIMEOUT_SECONDS:-}"
BATCH_STARTUP_TIMEOUT_SECONDS="${BATCH_STARTUP_TIMEOUT_SECONDS:-}"
BATCH_RESUME_PASS_MODE="${BATCH_RESUME_PASS_MODE:-}"
BATCH_RETRY_TIMEOUT_FAILURES_ON_RESUME="${BATCH_RETRY_TIMEOUT_FAILURES_ON_RESUME:-}"
BATCH_RETRY_OOM_FAILURES_ON_RESUME="${BATCH_RETRY_OOM_FAILURES_ON_RESUME:-}"
BATCH_RETRY_INFRASTRUCTURE_FAILURES_ON_RESUME="${BATCH_RETRY_INFRASTRUCTURE_FAILURES_ON_RESUME:-}"
BATCH_RETRY_STRUCTURAL_FAILURES_ON_RESUME="${BATCH_RETRY_STRUCTURAL_FAILURES_ON_RESUME:-}"
BATCH_WORKER_PROCESS_MODE="${BATCH_WORKER_PROCESS_MODE:-}"
BATCH_MAX_REQUESTS_PER_WORKER_PROCESS="${BATCH_MAX_REQUESTS_PER_WORKER_PROCESS:-}"
REMOTE_RUNTIME_MODE="${REMOTE_RUNTIME_MODE:-auto}"
REMOTE_DOCKER_IMAGE="${REMOTE_DOCKER_IMAGE:-}"
REMOTE_DOCKER_CONTAINER_NAME="${REMOTE_DOCKER_CONTAINER_NAME:-lcm-$(basename "$REMOTE_RESULTS_DIR")}"
REMOTE_DOCKER_PULL="${REMOTE_DOCKER_PULL:-1}"
LOCAL_RESULTS_SYNC_INTERVAL_SECONDS="${LOCAL_RESULTS_SYNC_INTERVAL_SECONDS:-60}"
LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS="${LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS:-600}"
LOCAL_RESULTS_SYNC_EXCLUDES="${LOCAL_RESULTS_SYNC_EXCLUDES:-}"
LOCAL_RESULTS_SYNC_LOG_PATH="${LOCAL_RESULTS_SYNC_LOG_PATH:-}"
LOCAL_RESULTS_SYNC_PID_PATH="${LOCAL_RESULTS_SYNC_PID_PATH:-}"
LOCAL_RESULTS_SYNC_STATUS_PATH="${LOCAL_RESULTS_SYNC_STATUS_PATH:-}"
LOCAL_RESULTS_SYNC_LAST_SUCCESS_PATH="${LOCAL_RESULTS_SYNC_LAST_SUCCESS_PATH:-}"
REMOTE_PREVIEW_SCRIPT="${REMOTE_PREVIEW_SCRIPT:-$REMOTE_REPO_DIR/scripts/vast/remote_resume_preview.sh}"
REMOTE_LAUNCH_SCRIPT="${REMOTE_LAUNCH_SCRIPT:-$REMOTE_REPO_DIR/scripts/vast/remote_resume_launch.sh}"
REMOTE_RUNTIME_DOCTOR_SCRIPT="${REMOTE_RUNTIME_DOCTOR_SCRIPT:-$REMOTE_REPO_DIR/scripts/vast/remote_runtime_doctor.sh}"
REMOTE_RUNTIME_MODE="$(vast_select_remote_runtime_mode "$REMOTE_RUNTIME_MODE" "$REMOTE_DOCKER_IMAGE")"
BATCH_EXCLUDED_DECODING_LABELS="${BATCH_EXCLUDED_DECODING_LABELS:-}"
REMOTE_ROOT_NAME="$(basename "$REMOTE_RESULTS_DIR")"
if [ -z "$BATCH_EXCLUDED_DECODING_LABELS" ]; then
  BATCH_EXCLUDED_DECODING_LABELS="$(
    python3 - <<'PY' "$LOCAL_REPO_DIR" "$REMOTE_ROOT_NAME"
from pathlib import Path
import sys

repo_root = Path(sys.argv[1])
sys.path.insert(0, str(repo_root / "src"))

from llm_conceptual_modeling.hf_resume_profile import resolve_resume_profile

profile = resolve_resume_profile(sys.argv[2])
print(",".join(profile.excluded_decoding_labels))
PY
  )"
fi
REMOTE_PREVIEW_ENV_PREFIX="BATCH_GENERATION_TIMEOUT_SECONDS='$BATCH_GENERATION_TIMEOUT_SECONDS' BATCH_STARTUP_TIMEOUT_SECONDS='$BATCH_STARTUP_TIMEOUT_SECONDS' BATCH_RESUME_PASS_MODE='$BATCH_RESUME_PASS_MODE' BATCH_RETRY_TIMEOUT_FAILURES_ON_RESUME='$BATCH_RETRY_TIMEOUT_FAILURES_ON_RESUME' BATCH_RETRY_OOM_FAILURES_ON_RESUME='$BATCH_RETRY_OOM_FAILURES_ON_RESUME' BATCH_RETRY_INFRASTRUCTURE_FAILURES_ON_RESUME='$BATCH_RETRY_INFRASTRUCTURE_FAILURES_ON_RESUME' BATCH_RETRY_STRUCTURAL_FAILURES_ON_RESUME='$BATCH_RETRY_STRUCTURAL_FAILURES_ON_RESUME' BATCH_WORKER_PROCESS_MODE='$BATCH_WORKER_PROCESS_MODE' BATCH_MAX_REQUESTS_PER_WORKER_PROCESS='$BATCH_MAX_REQUESTS_PER_WORKER_PROCESS' BATCH_EXCLUDED_DECODING_LABELS='$BATCH_EXCLUDED_DECODING_LABELS'"

SSH_CMD=($(vast_ssh_command "$SSH_PORT" "$SSH_KEY_PATH"))
RSYNC_SSH="$(vast_rsync_ssh_command "$SSH_PORT" "$SSH_KEY_PATH")"

resolve_local_config_source() {
  local config_path="$1"
  if [ -f "$config_path" ]; then
    printf '%s\n' "$config_path"
    return 0
  fi
  if vast_has_value "$LOCAL_RESULTS_DIR" && [ -f "$LOCAL_RESULTS_DIR/$config_path" ]; then
    printf '%s\n' "$LOCAL_RESULTS_DIR/$config_path"
    return 0
  fi
  if [ -f "$LOCAL_REPO_DIR/$config_path" ]; then
    printf '%s\n' "$LOCAL_REPO_DIR/$config_path"
    return 0
  fi
  return 1
}

LOCAL_CONFIG_SOURCE_PATH="$(resolve_local_config_source "$CONFIG_PATH" || true)"
if ! vast_has_value "$LOCAL_CONFIG_SOURCE_PATH"; then
  echo "Could not resolve config source path: $CONFIG_PATH" >&2
  exit 1
fi

case "$LOCAL_CONFIG_SOURCE_PATH" in
  "$LOCAL_RESULTS_DIR"/*)
    if ! vast_has_value "$LOCAL_RESULTS_DIR"; then
      echo "Config source lives under local results, but LOCAL_RESULTS_DIR was not provided." >&2
      exit 1
    fi
    REMOTE_CONFIG_SUFFIX="${LOCAL_CONFIG_SOURCE_PATH#$LOCAL_RESULTS_DIR/}"
    REMOTE_CONFIG_PATH="$REMOTE_RESULTS_DIR/$REMOTE_CONFIG_SUFFIX"
    ;;
  "$LOCAL_REPO_DIR"/*)
    REMOTE_CONFIG_SUFFIX="${LOCAL_CONFIG_SOURCE_PATH#$LOCAL_REPO_DIR/}"
    REMOTE_CONFIG_PATH="$REMOTE_REPO_DIR/$REMOTE_CONFIG_SUFFIX"
    ;;
  *)
    echo "Config source must live under the repo root or local results root: $LOCAL_CONFIG_SOURCE_PATH" >&2
    exit 1
    ;;
esac

echo "[0/6] Local resume preflight"
if vast_has_value "$LOCAL_RESULTS_DIR"; then
  uv --directory "$LOCAL_REPO_DIR" run lcm run resume-preflight \
    --config "$LOCAL_CONFIG_SOURCE_PATH" \
    --repo-root "$LOCAL_REPO_DIR" \
    --results-root "$LOCAL_RESULTS_DIR" \
    --json
else
  uv --directory "$LOCAL_REPO_DIR" run lcm run resume-preflight \
    --config "$LOCAL_CONFIG_SOURCE_PATH" \
    --repo-root "$LOCAL_REPO_DIR" \
    --allow-empty \
    --json
fi

echo "[1/6] Sync repository"
vast_retry_rsync 3 rsync -avz \
  -e "$RSYNC_SSH" \
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
  "$LOCAL_REPO_DIR"/ "$SSH_TARGET:$REMOTE_REPO_DIR"/

echo "[2/6] Validate remote runtime"
if [ "$REMOTE_RUNTIME_MODE" = "docker" ]; then
  "${SSH_CMD[@]}" "$SSH_TARGET" "REMOTE_RUNTIME_MODE='$REMOTE_RUNTIME_MODE' REMOTE_DOCKER_IMAGE='$REMOTE_DOCKER_IMAGE' bash '$REMOTE_RUNTIME_DOCTOR_SCRIPT' '$REMOTE_REPO_DIR'"
elif [ "$REMOTE_RUNTIME_MODE" = "bootstrap" ]; then
  "${SSH_CMD[@]}" "$SSH_TARGET" "REMOTE_RUNTIME_MODE='$REMOTE_RUNTIME_MODE' bash '$REMOTE_RUNTIME_DOCTOR_SCRIPT' '$REMOTE_REPO_DIR'"
else
  echo "Unsupported REMOTE_RUNTIME_MODE: $REMOTE_RUNTIME_MODE" >&2
  exit 1
fi

if vast_has_value "$LOCAL_RESULTS_DIR"; then
  echo "[3/6] Seed remote results root from local copy"
  mkdir -p "$LOCAL_RESULTS_DIR"
  "${SSH_CMD[@]}" "$SSH_TARGET" "mkdir -p '$REMOTE_RESULTS_DIR'"
  vast_retry_rsync 3 rsync -avz \
    $(vast_rsync_resume_flags "$LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS") \
    -e "$RSYNC_SSH" "$LOCAL_RESULTS_DIR"/ "$SSH_TARGET:$REMOTE_RESULTS_DIR"/
else
  echo "[3/6] No local results seed provided; skipping"
fi

echo "[4/6] Bootstrap remote runtime"
if [ "$REMOTE_RUNTIME_MODE" = "docker" ]; then
  if ! vast_has_value "$REMOTE_DOCKER_IMAGE"; then
    echo "REMOTE_DOCKER_IMAGE must be set when REMOTE_RUNTIME_MODE=docker" >&2
    exit 1
  fi
  "${SSH_CMD[@]}" "$SSH_TARGET" "
    set -euo pipefail
    command -v docker >/dev/null 2>&1
    if ! docker image inspect '$REMOTE_DOCKER_IMAGE' >/dev/null 2>&1; then
      if [ '$REMOTE_DOCKER_PULL' = '1' ]; then
        docker pull '$REMOTE_DOCKER_IMAGE'
      else
        echo 'Docker image not available: $REMOTE_DOCKER_IMAGE' >&2
        exit 1
      fi
    fi
    docker rm -f '$REMOTE_DOCKER_CONTAINER_NAME' >/dev/null 2>&1 || true
    docker run -d \
      --name '$REMOTE_DOCKER_CONTAINER_NAME' \
      --gpus all \
      -v '$REMOTE_REPO_DIR':'$REMOTE_REPO_DIR' \
      -v '$REMOTE_RESULTS_DIR':'$REMOTE_RESULTS_DIR' \
      -w '$REMOTE_REPO_DIR' \
      '$REMOTE_DOCKER_IMAGE' \
      tail -f /dev/null
  "
elif [ "$REMOTE_RUNTIME_MODE" = "bootstrap" ]; then
  "${SSH_CMD[@]}" "$SSH_TARGET" "bash '$REMOTE_REPO_DIR/scripts/vast/bootstrap_gpu_host.sh' '$REMOTE_REPO_DIR'"
else
  echo "Unsupported REMOTE_RUNTIME_MODE: $REMOTE_RUNTIME_MODE" >&2
  exit 1
fi

echo "[5/6] Run doctor + config preview"
if [ "$REMOTE_RUNTIME_MODE" = "docker" ]; then
  "${SSH_CMD[@]}" "$SSH_TARGET" "$REMOTE_PREVIEW_ENV_PREFIX docker exec '$REMOTE_DOCKER_CONTAINER_NAME' bash '$REMOTE_PREVIEW_SCRIPT' '$REMOTE_REPO_DIR' '$REMOTE_RESULTS_DIR' '$REMOTE_CONFIG_PATH' '$REMOTE_EFFECTIVE_CONFIG_PATH' '$REMOTE_PREVIEW_DIR'"
elif [ "$REMOTE_RUNTIME_MODE" = "bootstrap" ]; then
  "${SSH_CMD[@]}" "$SSH_TARGET" "$REMOTE_PREVIEW_ENV_PREFIX bash '$REMOTE_PREVIEW_SCRIPT' '$REMOTE_REPO_DIR' '$REMOTE_RESULTS_DIR' '$REMOTE_CONFIG_PATH' '$REMOTE_EFFECTIVE_CONFIG_PATH' '$REMOTE_PREVIEW_DIR'"
else
  echo "Unsupported REMOTE_RUNTIME_MODE: $REMOTE_RUNTIME_MODE" >&2
  exit 1
fi

if vast_has_value "${SMOKE_ALGORITHM:-}" \
  && vast_has_value "${SMOKE_MODEL:-}" \
  && vast_has_value "${SMOKE_PAIR_NAME:-}" \
  && vast_has_value "${SMOKE_CONDITION_BITS:-}" \
  && vast_has_value "${SMOKE_DECODING:-}"; then
  echo "[6/6] Run smoke gate"
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
  echo "[6/6] Smoke gate not configured; skipping"
fi

echo "[7/7] Launch resumable batch"
if [ "$REMOTE_RUNTIME_MODE" = "docker" ]; then
  "${SSH_CMD[@]}" "$SSH_TARGET" "docker exec '$REMOTE_DOCKER_CONTAINER_NAME' bash '$REMOTE_LAUNCH_SCRIPT' '$REMOTE_REPO_DIR' '$REMOTE_RESULTS_DIR' '$REMOTE_EFFECTIVE_CONFIG_PATH' '$REMOTE_RUN_LOG' '$REMOTE_PID_PATH'"
elif [ "$REMOTE_RUNTIME_MODE" = "bootstrap" ]; then
  "${SSH_CMD[@]}" "$SSH_TARGET" "bash '$REMOTE_LAUNCH_SCRIPT' '$REMOTE_REPO_DIR' '$REMOTE_RESULTS_DIR' '$REMOTE_EFFECTIVE_CONFIG_PATH' '$REMOTE_RUN_LOG' '$REMOTE_PID_PATH'"
else
  echo "Unsupported REMOTE_RUNTIME_MODE: $REMOTE_RUNTIME_MODE" >&2
  exit 1
fi

if vast_has_value "$LOCAL_RESULTS_DIR"; then
  echo "[sync] Launch local result autosync"
  if [ -z "$LOCAL_RESULTS_SYNC_LOG_PATH" ]; then
    LOCAL_RESULTS_SYNC_LOG_PATH="$LOCAL_RESULTS_DIR/results-sync.log"
  fi
  if [ -z "$LOCAL_RESULTS_SYNC_PID_PATH" ]; then
    LOCAL_RESULTS_SYNC_PID_PATH="$LOCAL_RESULTS_DIR/results-sync.pid"
  fi
  if [ -z "$LOCAL_RESULTS_SYNC_STATUS_PATH" ]; then
    LOCAL_RESULTS_SYNC_STATUS_PATH="$LOCAL_RESULTS_DIR/results-sync-status.json"
  fi
  if [ -z "$LOCAL_RESULTS_SYNC_LAST_SUCCESS_PATH" ]; then
    LOCAL_RESULTS_SYNC_LAST_SUCCESS_PATH="$LOCAL_RESULTS_DIR/results-sync-last-success.txt"
  fi
  mkdir -p "$LOCAL_RESULTS_DIR"
  pkill -f "scripts/vast/watch_results_from_vast.sh $SSH_TARGET:$REMOTE_RESULTS_DIR $LOCAL_RESULTS_DIR" || true
  LOCAL_RESULTS_SYNC_IDENTITY="$(vast_watcher_identity "$SSH_TARGET" "$SSH_PORT" "$REMOTE_RESULTS_DIR")"
  RSYNC_TIMEOUT_SECONDS="$LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS" \
  LOCAL_RESULTS_SYNC_EXCLUDES="$LOCAL_RESULTS_SYNC_EXCLUDES" \
  SSH_KEY_PATH="$SSH_KEY_PATH" \
  SSH_PORT="$SSH_PORT" \
  SYNC_INTERVAL_SECONDS="$LOCAL_RESULTS_SYNC_INTERVAL_SECONDS" \
  LOCAL_RESULTS_SYNC_IDENTITY="$LOCAL_RESULTS_SYNC_IDENTITY" \
  LOCAL_RESULTS_SYNC_STATUS_PATH="$LOCAL_RESULTS_SYNC_STATUS_PATH" \
  LOCAL_RESULTS_SYNC_LAST_SUCCESS_PATH="$LOCAL_RESULTS_SYNC_LAST_SUCCESS_PATH" \
  nohup bash "$LOCAL_REPO_DIR/scripts/vast/watch_results_from_vast.sh" \
    "$SSH_TARGET:$REMOTE_RESULTS_DIR" \
    "$LOCAL_RESULTS_DIR" \
    "$SSH_PORT" \
    >"$LOCAL_RESULTS_SYNC_LOG_PATH" 2>&1 </dev/null &
  echo $! > "$LOCAL_RESULTS_SYNC_PID_PATH"
fi
