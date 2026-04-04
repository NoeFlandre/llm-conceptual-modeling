#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
  cat >&2 <<'USAGE'
usage: drain_qwen_batches_from_ssh.sh SSH_COMMAND [LOCAL_REPO_DIR] [LOCAL_RESULTS_DIR]

Example:
  scripts/vast/drain_qwen_batches_from_ssh.sh \
    "ssh -p 30444 root@59.7.51.43 -L 8080:localhost:8080" \
    /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling \
    /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/results
USAGE
  exit 1
fi

SSH_COMMAND="$1"
LOCAL_REPO_DIR="${2:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
LOCAL_RESULTS_DIR="${3:-$LOCAL_REPO_DIR/results}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_rsa}"
QWEN_QUICK_RESUME_SCRIPT="${QWEN_QUICK_RESUME_SCRIPT:-$SCRIPT_DIR/quick_resume_from_ssh.sh}"
QWEN_LAUNCH_ATTEMPTS="${QWEN_LAUNCH_ATTEMPTS:-3}"
QWEN_LAUNCH_BACKOFF_SECONDS="${QWEN_LAUNCH_BACKOFF_SECONDS:-15}"
QWEN_GENERATION_TIMEOUT_SECONDS="${QWEN_GENERATION_TIMEOUT_SECONDS:-60}"
QWEN_RETRY_OOM_FAILURES_ON_RESUME="${QWEN_RETRY_OOM_FAILURES_ON_RESUME:-true}"
QWEN_RETRY_STRUCTURAL_FAILURES_ON_RESUME="${QWEN_RETRY_STRUCTURAL_FAILURES_ON_RESUME:-true}"
QWEN_RETRY_TIMEOUT_FAILURES_ON_RESUME="${QWEN_RETRY_TIMEOUT_FAILURES_ON_RESUME:-true}"
QWEN_RETRY_INFRASTRUCTURE_FAILURES_ON_RESUME="${QWEN_RETRY_INFRASTRUCTURE_FAILURES_ON_RESUME:-true}"
QWEN_LOCAL_RESULTS_SYNC_INTERVAL_SECONDS="${QWEN_LOCAL_RESULTS_SYNC_INTERVAL_SECONDS:-30}"
QWEN_LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS="${QWEN_LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS:-300}"
QWEN_DRAIN_MAX_PASSES="${QWEN_DRAIN_MAX_PASSES:-0}"
QWEN_RUNNING_ROOT_STALE_SECONDS="${QWEN_RUNNING_ROOT_STALE_SECONDS:-3600}"
QWEN_ROOTS=(
  "hf-paper-batch-algo1-qwen"
  "hf-paper-batch-algo2-qwen-current"
  "hf-paper-batch-algo3-qwen-current"
)
QWEN_DRAIN_POLL_SECONDS="${QWEN_DRAIN_POLL_SECONDS:-30}"
QWEN_DRAIN_MAX_WAIT_SECONDS="${QWEN_DRAIN_MAX_WAIT_SECONDS:-0}"

read_ssh_target() {
  SSH_COMMAND="$SSH_COMMAND" python3 - <<'PY'
import os
import shlex

tokens = shlex.split(os.environ["SSH_COMMAND"])
target = None
port = None
index = 0
while index < len(tokens):
    token = tokens[index]
    if token == "ssh":
        index += 1
        continue
    if token == "-p" and index + 1 < len(tokens):
        port = tokens[index + 1]
        index += 2
        continue
    if token.startswith("-"):
        index += 1
        continue
    target = token
    break

if not target or not port:
    raise SystemExit("Could not parse SSH target and port from SSH_COMMAND")

print(target)
print(port)
PY
}

PARSED_SSH="$(read_ssh_target)"
SSH_TARGET="$(printf '%s\n' "$PARSED_SSH" | sed -n '1p')"
SSH_PORT="$(printf '%s\n' "$PARSED_SSH" | sed -n '2p')"

root_results_dir() {
  local root_name="$1"
  printf '%s/%s' "$LOCAL_RESULTS_DIR" "$root_name"
}

root_status_updated_at_epoch() {
  local root_name="$1"
  local updated_at
  local status_path
  status_path="$(root_results_dir "$root_name")/batch_status.json"
  if [ ! -f "$status_path" ]; then
    printf '%s\n' "-1"
    return 0
  fi
  updated_at="$(python3 - <<'PY' "$status_path"
import json
import sys
from pathlib import Path

try:
    status = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except Exception:
    print("")
else:
    print(status.get("updated_at", ""))
PY
)"
  if [ -z "$updated_at" ]; then
    printf '%s\n' "-1"
    return 0
  fi
  python3 - <<'PY' "$updated_at"
from datetime import datetime, timezone
import sys

value = sys.argv[1].strip()
if value.endswith("Z"):
    value = value[:-1] + "+00:00"
try:
    parsed = datetime.fromisoformat(value)
except ValueError:
    print(-1)
    raise SystemExit(0)
if parsed.tzinfo is None:
    parsed = parsed.replace(tzinfo=timezone.utc)
print(int(parsed.timestamp()))
PY
}

root_status_is_stale() {
  local root_name="$1"
  local updated_at_epoch now age_seconds
  if [ "$QWEN_RUNNING_ROOT_STALE_SECONDS" -le 0 ]; then
    return 0
  fi
  updated_at_epoch="$(root_status_updated_at_epoch "$root_name")"
  if [ "$updated_at_epoch" -lt 0 ]; then
    return 0
  fi
  now="$(date +%s)"
  age_seconds="$((now - updated_at_epoch))"
  [ "$age_seconds" -ge "$QWEN_RUNNING_ROOT_STALE_SECONDS" ]
}

root_config_source() {
  local root_name="$1"
  local root_dir
  local repo_config
  root_dir="$(root_results_dir "$root_name")"
  for candidate in \
    "$root_dir/preview_resume/resolved_run_config.yaml" \
    "$root_dir/runtime_config.yaml" \
    "$root_dir/preview/resolved_run_config.yaml"; do
    if [ -f "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  case "$root_name" in
    hf-paper-batch-algo2-qwen-current) repo_config="$LOCAL_REPO_DIR/configs/hf_transformers_algo2_qwen.yaml" ;;
    hf-paper-batch-algo3-qwen-current) repo_config="$LOCAL_REPO_DIR/configs/hf_transformers_algo3_qwen.yaml" ;;
    *)
      repo_config=""
      ;;
  esac
  if [ -n "$repo_config" ] && [ -f "$repo_config" ]; then
    printf '%s\n' "$repo_config"
    return 0
  fi
  return 1
}

root_excluded_decoding_labels() {
  _="${1:-}"
  printf '%s\n' ""
}

read_root_status() {
  local root_name="$1"
  local status_path
  status_path="$(root_results_dir "$root_name")/batch_status.json"
  STATUS_PATH="$status_path" python3 - <<'PY'
import json
import os
from pathlib import Path

status_path = Path(os.environ["STATUS_PATH"])
if not status_path.exists():
    print("0 0 0 0 missing")
    raise SystemExit(0)

try:
    status = json.loads(status_path.read_text(encoding="utf-8"))
except Exception:
    print("0 0 0 0 corrupt")
else:
    print(
        f"{int(status.get('finished_count', 0))} "
        f"{int(status.get('pending_count', 0))} "
        f"{int(status.get('failed_count', 0))} "
        f"{int(status.get('running_count', 0))} "
        f"{status.get('updated_at', 'unknown')}"
    )
PY
}

wait_for_root_completion() {
  local root_name="$1"
  local started_at now elapsed
  started_at="$(date +%s)"
  while true; do
    local status_line finished pending failed running updated_at
    status_line="$(read_root_status "$root_name")"
    finished="$(printf '%s\n' "$status_line" | awk '{print $1}')"
    pending="$(printf '%s\n' "$status_line" | awk '{print $2}')"
    failed="$(printf '%s\n' "$status_line" | awk '{print $3}')"
    running="$(printf '%s\n' "$status_line" | awk '{print $4}')"
    updated_at="$(printf '%s\n' "$status_line" | awk '{print $5}')"

    echo "[$root_name] finished=$finished pending=$pending failed=$failed running=$running updated_at=$updated_at"

    if [ "$pending" = "0" ] && [ "$running" = "0" ] && [ "$failed" = "0" ]; then
      return 0
    fi

    if [ "$QWEN_DRAIN_MAX_WAIT_SECONDS" -gt 0 ]; then
      now="$(date +%s)"
      elapsed="$((now - started_at))"
      if [ "$elapsed" -ge "$QWEN_DRAIN_MAX_WAIT_SECONDS" ]; then
        echo "Timed out waiting for $root_name after ${QWEN_DRAIN_MAX_WAIT_SECONDS}s" >&2
        return 1
      fi
    fi

    sleep "$QWEN_DRAIN_POLL_SECONDS"
  done
}

launch_root_resume() {
  local root_name="$1"
  local config_path
  local remote_results_dir
  local local_results_dir
  local excluded_decoding_labels
  local attempt status

  config_path="$(root_config_source "$root_name")"
  remote_results_dir="/workspace/results/$root_name"
  local_results_dir="$(root_results_dir "$root_name")"
  excluded_decoding_labels="$(root_excluded_decoding_labels "$root_name")"

  if [ -z "$config_path" ]; then
    echo "Missing local resume config for $root_name" >&2
    return 1
  fi

  attempt=1
  while true; do
    echo "[launch] $root_name attempt $attempt/$QWEN_LAUNCH_ATTEMPTS"
    if BATCH_EXCLUDED_DECODING_LABELS="$excluded_decoding_labels" \
      LOCAL_RESULTS_SYNC_INTERVAL_SECONDS="$QWEN_LOCAL_RESULTS_SYNC_INTERVAL_SECONDS" \
      LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS="$QWEN_LOCAL_RESULTS_SYNC_RSYNC_TIMEOUT_SECONDS" \
      BATCH_GENERATION_TIMEOUT_SECONDS="$QWEN_GENERATION_TIMEOUT_SECONDS" \
      BATCH_RETRY_TIMEOUT_FAILURES_ON_RESUME="$QWEN_RETRY_TIMEOUT_FAILURES_ON_RESUME" \
      BATCH_RETRY_INFRASTRUCTURE_FAILURES_ON_RESUME="$QWEN_RETRY_INFRASTRUCTURE_FAILURES_ON_RESUME" \
      BATCH_RETRY_OOM_FAILURES_ON_RESUME="$QWEN_RETRY_OOM_FAILURES_ON_RESUME" \
      BATCH_RETRY_STRUCTURAL_FAILURES_ON_RESUME="$QWEN_RETRY_STRUCTURAL_FAILURES_ON_RESUME" \
      "$QWEN_QUICK_RESUME_SCRIPT" \
      "$SSH_COMMAND" \
      "$config_path" \
      "$remote_results_dir" \
      "$local_results_dir"; then
      return 0
    fi
    status=$?
    if [ "$attempt" -ge "$QWEN_LAUNCH_ATTEMPTS" ]; then
      return "$status"
    fi
    echo "[retry] $root_name launch failed; backing off for ${QWEN_LAUNCH_BACKOFF_SECONDS}s"
    sleep "$QWEN_LAUNCH_BACKOFF_SECONDS"
    attempt=$((attempt + 1))
  done
}

read_preflight_report() {
  local root_name="$1"
  local config_path
  local local_results_dir
  config_path="$(root_config_source "$root_name")"
  if [ -z "$config_path" ]; then
    echo "Missing local resume config for $root_name" >&2
    return 1
  fi
  local_results_dir="$(root_results_dir "$root_name")"

  uv --directory "$LOCAL_REPO_DIR" run lcm run resume-preflight \
    --config "$config_path" \
    --repo-root "$LOCAL_REPO_DIR" \
    --results-root "$local_results_dir" \
    --json
}

qwen_work_remaining() {
  local root_name
  for root_name in "${QWEN_ROOTS[@]}"; do
    local root_dir status_line pending running failed
    root_dir="$(root_results_dir "$root_name")"
    if [ ! -d "$root_dir" ]; then
      continue
    fi
    status_line="$(read_root_status "$root_name")"
    pending="$(printf '%s\n' "$status_line" | awk '{print $2}')"
    running="$(printf '%s\n' "$status_line" | awk '{print $4}')"
    if [ "$pending" != "0" ] || [ "$running" != "0" ]; then
      return 0
    fi
    failed="$(printf '%s\n' "$status_line" | awk '{print $3}')"
    if [ "$failed" != "0" ] && root_can_resume_from_preflight "$root_name"; then
      return 0
    fi
  done
  return 1
}

root_can_resume_from_preflight() {
  local root_name="$1"
  local report_json can_resume pending failed
  if ! report_json="$(read_preflight_report "$root_name")"; then
    return 1
  fi
  can_resume="$(REPORT_JSON="$report_json" python3 - <<'PY'
import json
import os

data = json.loads(os.environ["REPORT_JSON"])
print("true" if data.get("can_resume") else "false")
PY
)"
  if [ "$can_resume" != "true" ]; then
    return 1
  fi
  pending="$(REPORT_JSON="$report_json" python3 - <<'PY'
import json
import os

data = json.loads(os.environ["REPORT_JSON"])
print(int(data.get("pending_count", 0)))
PY
)"
  failed="$(REPORT_JSON="$report_json" python3 - <<'PY'
import json
import os

data = json.loads(os.environ["REPORT_JSON"])
print(int(data.get("failed_count", 0)))
PY
)"
  [ "$pending" != "0" ] || [ "$failed" != "0" ]
}

mkdir -p "$LOCAL_RESULTS_DIR"

pass_index=1
while true; do
  echo "[pass $pass_index] scanning QWEN roots"

  for root_name in "${QWEN_ROOTS[@]}"; do
    root_dir="$(root_results_dir "$root_name")"
    if [ ! -d "$root_dir" ]; then
      echo "Skipping missing root: $root_dir"
      continue
    fi

    status_line="$(read_root_status "$root_name")"
    finished="$(printf '%s\n' "$status_line" | awk '{print $1}')"
    pending="$(printf '%s\n' "$status_line" | awk '{print $2}')"
    failed="$(printf '%s\n' "$status_line" | awk '{print $3}')"
    running="$(printf '%s\n' "$status_line" | awk '{print $4}')"
    updated_at="$(printf '%s\n' "$status_line" | awk '{print $5}')"

    echo "[$root_name] finished=$finished pending=$pending failed=$failed running=$running updated_at=$updated_at"

    if [ "$pending" = "0" ] && [ "$running" = "0" ]; then
      if ! root_can_resume_from_preflight "$root_name"; then
        continue
      fi
      echo "[resume] $root_name has retryable failed work according to preflight"
    fi

    if [ "$running" != "0" ]; then
      if root_status_is_stale "$root_name"; then
        echo "[stale] $root_name running status is stale; reclaiming root"
      else
        echo "[wait] $root_name is already active; waiting for completion"
        wait_for_root_completion "$root_name"
        continue
      fi
    fi

    launch_root_resume "$root_name"
    wait_for_root_completion "$root_name"
  done

  if ! qwen_work_remaining; then
    echo "No pending or running QWEN work remains."
    break
  fi

  if [ "$QWEN_DRAIN_MAX_PASSES" -gt 0 ] && [ "$pass_index" -ge "$QWEN_DRAIN_MAX_PASSES" ]; then
    echo "Reached configured maximum pass count: $QWEN_DRAIN_MAX_PASSES"
    break
  fi

  pass_index=$((pass_index + 1))
done
