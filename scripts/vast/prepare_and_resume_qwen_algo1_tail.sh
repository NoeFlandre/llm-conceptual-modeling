#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

if [ "$#" -ne 7 ]; then
  cat >&2 <<'USAGE'
usage: prepare_and_resume_qwen_algo1_tail.sh SSH_TARGET SSH_PORT LOCAL_REPO_DIR REMOTE_REPO_DIR CANONICAL_RESULTS_ROOT LOCAL_TAIL_RESULTS_ROOT REMOTE_TAIL_RESULTS_ROOT
USAGE
  exit 1
fi

SSH_TARGET="$1"
SSH_PORT="$2"
LOCAL_REPO_DIR="$3"
REMOTE_REPO_DIR="$4"
CANONICAL_RESULTS_ROOT="$5"
LOCAL_TAIL_RESULTS_ROOT="$6"
REMOTE_TAIL_RESULTS_ROOT="$7"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_rsa}"
REMOTE_RUN_LOG="${REMOTE_RUN_LOG:-$REMOTE_TAIL_RESULTS_ROOT/run.log}"
REMOTE_PID_PATH="${REMOTE_PID_PATH:-$REMOTE_TAIL_RESULTS_ROOT/batch.pid}"
REMOTE_PREVIEW_DIR="${REMOTE_PREVIEW_DIR:-$REMOTE_TAIL_RESULTS_ROOT/preview}"
LOCAL_RESULTS_SYNC_INTERVAL_SECONDS="${LOCAL_RESULTS_SYNC_INTERVAL_SECONDS:-60}"
LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS="${LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS:-600}"
LOCAL_RESULTS_SYNC_LOG_PATH="${LOCAL_RESULTS_SYNC_LOG_PATH:-$LOCAL_TAIL_RESULTS_ROOT/watch.log}"
LOCAL_RESULTS_SYNC_PID_PATH="${LOCAL_RESULTS_SYNC_PID_PATH:-$LOCAL_TAIL_RESULTS_ROOT/watch.pid}"
LOCAL_RESULTS_SYNC_STATUS_PATH="${LOCAL_RESULTS_SYNC_STATUS_PATH:-$LOCAL_TAIL_RESULTS_ROOT/results-sync-status.json}"
LOCAL_RESULTS_SYNC_LAST_SUCCESS_PATH="${LOCAL_RESULTS_SYNC_LAST_SUCCESS_PATH:-$LOCAL_TAIL_RESULTS_ROOT/results-sync-last-success.txt}"
LOCAL_RESULTS_SYNC_IDENTITY="${LOCAL_RESULTS_SYNC_IDENTITY:-$(vast_watcher_identity "$SSH_TARGET" "$SSH_PORT" "$REMOTE_TAIL_RESULTS_ROOT")}"

SSH_CMD=($(vast_ssh_command "$SSH_PORT" "$SSH_KEY_PATH"))
RSYNC_SSH="$(vast_rsync_ssh_command "$SSH_PORT" "$SSH_KEY_PATH")"

echo "[1/6] Materialize dedicated Qwen algo1 tail"
uv --directory "$LOCAL_REPO_DIR" run lcm run prepare-qwen-algo1-tail \
  --canonical-results-root "$CANONICAL_RESULTS_ROOT" \
  --tail-results-root "$LOCAL_TAIL_RESULTS_ROOT" \
  --remote-output-root "$REMOTE_TAIL_RESULTS_ROOT" \
  --json

echo "[2/6] Local preflight"
uv --directory "$LOCAL_REPO_DIR" run lcm run qwen-algo1-tail-preflight \
  --repo-root "$LOCAL_REPO_DIR" \
  --canonical-results-root "$CANONICAL_RESULTS_ROOT" \
  --tail-results-root "$LOCAL_TAIL_RESULTS_ROOT" \
  --json

echo "[3/6] Sync repository"
vast_retry_rsync 3 rsync -avz \
  --delete \
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
  "$LOCAL_REPO_DIR"/ "$SSH_TARGET:$REMOTE_REPO_DIR"/

echo "[4/6] Sync dedicated tail root"
"${SSH_CMD[@]}" "$SSH_TARGET" "mkdir -p '$REMOTE_TAIL_RESULTS_ROOT'"
vast_retry_rsync 3 rsync -avz \
  $(vast_rsync_resume_flags "$LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS") \
  -e "$RSYNC_SSH" \
  "$LOCAL_TAIL_RESULTS_ROOT"/ "$SSH_TARGET:$REMOTE_TAIL_RESULTS_ROOT"/

echo "[5/6] Bootstrap and validate remote runtime"
"${SSH_CMD[@]}" "$SSH_TARGET" "bash '$REMOTE_REPO_DIR/scripts/vast/bootstrap_gpu_host.sh' '$REMOTE_REPO_DIR'"
"${SSH_CMD[@]}" "$SSH_TARGET" "bash '$REMOTE_REPO_DIR/scripts/vast/remote_runtime_doctor.sh' '$REMOTE_REPO_DIR'"
"${SSH_CMD[@]}" "$SSH_TARGET" "bash '$REMOTE_REPO_DIR/scripts/vast/remote_resume_preview.sh' '$REMOTE_REPO_DIR' '$REMOTE_TAIL_RESULTS_ROOT' '$REMOTE_TAIL_RESULTS_ROOT/runtime_config.yaml' '$REMOTE_TAIL_RESULTS_ROOT/runtime_config.yaml' '$REMOTE_PREVIEW_DIR'"

echo "[6/6] Launch dedicated Qwen algo1 tail batch"
"${SSH_CMD[@]}" "$SSH_TARGET" "
  set -euo pipefail
  mkdir -p '$REMOTE_TAIL_RESULTS_ROOT'
  nohup bash -lc '
    set -euo pipefail
    cd \"$REMOTE_REPO_DIR\"
    export PYTHONPATH=\"$REMOTE_REPO_DIR/src\${PYTHONPATH:+:\$PYTHONPATH}\"
    while true; do
      env -u NVIDIA_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
        \"$REMOTE_REPO_DIR/.venv/bin/lcm\" run paper-batch --config \"$REMOTE_TAIL_RESULTS_ROOT/runtime_config.yaml\" --resume || true
      pending_count=\$(python3 - <<\"PY\"
from pathlib import Path
import json
path = Path(\"$REMOTE_TAIL_RESULTS_ROOT\") / \"batch_status.json\"
if not path.exists():
    print(0)
else:
    payload = json.loads(path.read_text(encoding=\"utf-8\"))
    print(int(payload.get(\"pending_count\", 0)))
PY
)
      if [ \"\$pending_count\" -le 0 ]; then
        break
      fi
      sleep 5
    done
  ' </dev/null >>'$REMOTE_RUN_LOG' 2>&1 &
  echo \$! > '$REMOTE_PID_PATH'
  cat '$REMOTE_PID_PATH'
"

echo "Start dedicated local watcher"
mkdir -p "$LOCAL_TAIL_RESULTS_ROOT"
nohup bash "$LOCAL_REPO_DIR/scripts/vast/watch_results_from_vast.sh" \
  "$SSH_TARGET:$REMOTE_TAIL_RESULTS_ROOT" \
  "$LOCAL_TAIL_RESULTS_ROOT" \
  "$SSH_PORT" \
  >"$LOCAL_RESULTS_SYNC_LOG_PATH" 2>&1 </dev/null &
echo $! > "$LOCAL_RESULTS_SYNC_PID_PATH"

echo "watcher_pid=$(cat "$LOCAL_RESULTS_SYNC_PID_PATH")"
echo "remote_pid_path=$REMOTE_PID_PATH"
