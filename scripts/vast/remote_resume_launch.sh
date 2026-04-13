#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

if [ "$#" -ne 5 ]; then
  cat >&2 <<'USAGE'
usage: remote_resume_launch.sh REMOTE_REPO_DIR REMOTE_RESULTS_DIR REMOTE_EFFECTIVE_CONFIG_PATH REMOTE_RUN_LOG REMOTE_PID_PATH
USAGE
  exit 1
fi

REMOTE_REPO_DIR="$1"
REMOTE_RESULTS_DIR="$2"
REMOTE_EFFECTIVE_CONFIG_PATH="$3"
REMOTE_RUN_LOG="$4"
REMOTE_PID_PATH="$5"
REMOTE_GPU_LIVENESS_TIMEOUT_SECONDS="${REMOTE_GPU_LIVENESS_TIMEOUT_SECONDS:-600}"
REMOTE_GPU_LIVENESS_POLL_INTERVAL_SECONDS="${REMOTE_GPU_LIVENESS_POLL_INTERVAL_SECONDS:-5}"
REMOTE_PRODUCTIVE_LIVENESS_TIMEOUT_SECONDS="${REMOTE_PRODUCTIVE_LIVENESS_TIMEOUT_SECONDS:-900}"
REMOTE_PRODUCTIVE_LIVENESS_POLL_INTERVAL_SECONDS="${REMOTE_PRODUCTIVE_LIVENESS_POLL_INTERVAL_SECONDS:-5}"
REMOTE_PROCESS_EXIT_TIMEOUT_SECONDS="${REMOTE_PROCESS_EXIT_TIMEOUT_SECONDS:-60}"
REMOTE_RELAUNCH_IF_PENDING="${REMOTE_RELAUNCH_IF_PENDING:-1}"
REMOTE_RELAUNCH_SLEEP_SECONDS="${REMOTE_RELAUNCH_SLEEP_SECONDS:-5}"

cd "$REMOTE_REPO_DIR"
mkdir -p "$REMOTE_RESULTS_DIR"
export PYTHONPATH="$REMOTE_REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

vast_wait_for_process_exit() {
  local pattern="$1"
  local deadline=$((SECONDS + REMOTE_PROCESS_EXIT_TIMEOUT_SECONDS))
  while pgrep -f -- "$pattern" >/dev/null 2>&1; do
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "process pattern failed to exit within ${REMOTE_PROCESS_EXIT_TIMEOUT_SECONDS}s: $pattern" >&2
      return 1
    fi
    sleep 1
  done
}

pkill -f -- ".venv/bin/lcm run paper-batch --config $REMOTE_EFFECTIVE_CONFIG_PATH --resume" || true
pkill -f -- "llm_conceptual_modeling.hf_worker --queue-dir $REMOTE_RESULTS_DIR/worker-queues/" || true
vast_wait_for_process_exit ".venv/bin/lcm run paper-batch --config $REMOTE_EFFECTIVE_CONFIG_PATH --resume"
vast_wait_for_process_exit "llm_conceptual_modeling.hf_worker --queue-dir $REMOTE_RESULTS_DIR/worker-queues/"
sleep 2

find "$REMOTE_RESULTS_DIR/worker-queues" -type f \
  \( -name '*.request.json' -o -name '*.claimed.json' -o -name '*.result.json' \) \
  -delete 2>/dev/null || true

vast_worker_pid() {
  pgrep -n -f 'llm_conceptual_modeling.hf_worker --queue-dir' || true
}

vast_worker_uses_gpu() {
  local worker_pid="$1"
  if [ -z "$worker_pid" ] || [ ! -d "/proc/$worker_pid" ]; then
    return 1
  fi
  if [ -f "/proc/$worker_pid/environ" ] \
    && tr '\0' '\n' < "/proc/$worker_pid/environ" | grep -qx 'NVIDIA_VISIBLE_DEVICES=void'; then
    echo "worker inherited NVIDIA_VISIBLE_DEVICES=void; refusing CPU-only resume" >&2
    return 1
  fi
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk '{print $1}' | grep -qx "$worker_pid"
}

vast_wait_for_gpu_liveness() {
  local deadline=$((SECONDS + REMOTE_GPU_LIVENESS_TIMEOUT_SECONDS))
  local worker_pid=""
  while [ "$SECONDS" -lt "$deadline" ]; do
    worker_pid="$(vast_worker_pid)"
    if vast_worker_uses_gpu "$worker_pid"; then
      return 0
    fi
    sleep "$REMOTE_GPU_LIVENESS_POLL_INTERVAL_SECONDS"
  done

  echo "worker failed to attach to GPU within ${REMOTE_GPU_LIVENESS_TIMEOUT_SECONDS}s" >&2
  return 1
}

vast_worker_is_executing_algorithm() {
  python3 - <<'PY' "$REMOTE_RESULTS_DIR/worker_state.json"
from pathlib import Path
import json
import sys

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding="utf-8"))
phase = payload.get("phase")
model_loaded = bool(payload.get("model_loaded"))
raise SystemExit(0 if model_loaded or phase == "executing_algorithm" else 1)
PY
}

vast_finished_count() {
  python3 - <<'PY' "$REMOTE_RESULTS_DIR/batch_status.json"
from pathlib import Path
import json
import sys

path = Path(sys.argv[1])
if not path.exists():
    print(0)
    raise SystemExit(0)
payload = json.loads(path.read_text(encoding="utf-8"))
print(int(payload.get("finished_count", 0)))
PY
}

vast_pending_count() {
  python3 - <<'PY' "$REMOTE_RESULTS_DIR/batch_status.json"
from pathlib import Path
import json
import sys

path = Path(sys.argv[1])
if not path.exists():
    print(0)
    raise SystemExit(0)
payload = json.loads(path.read_text(encoding="utf-8"))
print(int(payload.get("pending_count", 0)))
PY
}

vast_wait_for_productive_liveness() {
  local initial_finished_count
  local current_finished_count
  local deadline=$((SECONDS + REMOTE_PRODUCTIVE_LIVENESS_TIMEOUT_SECONDS))
  initial_finished_count="$(vast_finished_count)"

  while [ "$SECONDS" -lt "$deadline" ]; do
    if vast_worker_is_executing_algorithm; then
      return 0
    fi
    current_finished_count="$(vast_finished_count)"
    if [ "$current_finished_count" -gt "$initial_finished_count" ]; then
      return 0
    fi
    sleep "$REMOTE_PRODUCTIVE_LIVENESS_POLL_INTERVAL_SECONDS"
  done

  echo "batch failed to reach productive liveness within ${REMOTE_PRODUCTIVE_LIVENESS_TIMEOUT_SECONDS}s" >&2
  return 1
}

export REMOTE_EFFECTIVE_CONFIG_PATH
export REMOTE_RUN_LOG
export REMOTE_REPO_DIR
export REMOTE_RELAUNCH_IF_PENDING
export REMOTE_RELAUNCH_SLEEP_SECONDS
export REMOTE_RESULTS_DIR

nohup bash <<'EOF' </dev/null >>"$REMOTE_RUN_LOG" 2>&1 &
echo "[remote_resume_launch] starting supervised resume loop"
while true; do
  pkill -f "llm_conceptual_modeling.hf_worker --queue-dir ${REMOTE_RESULTS_DIR}/worker-queues/" >/dev/null 2>&1 || true
  echo "[remote_resume_launch] cleaned stale workers before resume"
  env -u NVIDIA_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
    PYTHONPATH="$REMOTE_REPO_DIR/src" \
    "$REMOTE_REPO_DIR/.venv/bin/lcm" run paper-batch --config "$REMOTE_EFFECTIVE_CONFIG_PATH" --resume || true
  if [ "$REMOTE_RELAUNCH_IF_PENDING" != "1" ]; then
    echo "[remote_resume_launch] relaunch disabled; exiting supervisor"
    break
  fi
  pending_count="$(python3 - <<'PY'
import os
from pathlib import Path
import json

path = Path(os.environ["REMOTE_RESULTS_DIR"]) / "batch_status.json"
if not path.exists():
    print(0)
else:
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(int(payload.get("pending_count", 0)))
PY
)"
  echo "[remote_resume_launch] pending_count=$pending_count"
  if [ "$pending_count" -le 0 ]; then
    echo "[remote_resume_launch] pending complete; exiting supervisor"
    break
  fi
  sleep "$REMOTE_RELAUNCH_SLEEP_SECONDS"
done
EOF
supervisor_pid=$!
sleep 2
echo "$supervisor_pid" > "$REMOTE_PID_PATH"
cat "$REMOTE_PID_PATH"

vast_wait_for_gpu_liveness
vast_wait_for_productive_liveness
