#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

if [ "$#" -ne 5 ]; then
  cat >&2 <<'USAGE'
usage: remote_resume_preview.sh REMOTE_REPO_DIR REMOTE_RESULTS_DIR REMOTE_CONFIG_PATH REMOTE_EFFECTIVE_CONFIG_PATH REMOTE_PREVIEW_DIR
USAGE
  exit 1
fi

REMOTE_REPO_DIR="$1"
REMOTE_RESULTS_DIR="$2"
REMOTE_CONFIG_PATH="$3"
REMOTE_EFFECTIVE_CONFIG_PATH="$4"
REMOTE_PREVIEW_DIR="$5"

cd "$REMOTE_REPO_DIR"
mkdir -p "$REMOTE_RESULTS_DIR"

export BATCH_GENERATION_TIMEOUT_SECONDS="${BATCH_GENERATION_TIMEOUT_SECONDS:-}"
export BATCH_STARTUP_TIMEOUT_SECONDS="${BATCH_STARTUP_TIMEOUT_SECONDS:-}"
export BATCH_RESUME_PASS_MODE="${BATCH_RESUME_PASS_MODE:-}"
export BATCH_RETRY_TIMEOUT_FAILURES_ON_RESUME="${BATCH_RETRY_TIMEOUT_FAILURES_ON_RESUME:-}"
export BATCH_RETRY_OOM_FAILURES_ON_RESUME="${BATCH_RETRY_OOM_FAILURES_ON_RESUME:-}"
export BATCH_RETRY_INFRASTRUCTURE_FAILURES_ON_RESUME="${BATCH_RETRY_INFRASTRUCTURE_FAILURES_ON_RESUME:-}"
export BATCH_RETRY_STRUCTURAL_FAILURES_ON_RESUME="${BATCH_RETRY_STRUCTURAL_FAILURES_ON_RESUME:-}"
export BATCH_WORKER_PROCESS_MODE="${BATCH_WORKER_PROCESS_MODE:-}"
export BATCH_MAX_REQUESTS_PER_WORKER_PROCESS="${BATCH_MAX_REQUESTS_PER_WORKER_PROCESS:-}"
export BATCH_EXCLUDED_DECODING_LABELS="${BATCH_EXCLUDED_DECODING_LABELS:-}"

.venv/bin/python - "$REMOTE_CONFIG_PATH" "$REMOTE_EFFECTIVE_CONFIG_PATH" <<'PY'
from pathlib import Path
import os
import sys

import yaml

sys.path.insert(0, str(Path.cwd() / "src"))

from llm_conceptual_modeling.hf_run_config import exclude_decoding_conditions_from_payload

source_path = Path(sys.argv[1])
target_path = Path(sys.argv[2])
payload = yaml.safe_load(source_path.read_text(encoding='utf-8'))
context_policy = dict(payload['runtime'].get('context_policy', {}))

timeout_value = os.environ.get('BATCH_GENERATION_TIMEOUT_SECONDS', '').strip()
if timeout_value:
    context_policy['generation_timeout_seconds'] = float(timeout_value)

timeout_value = os.environ.get('BATCH_STARTUP_TIMEOUT_SECONDS', '').strip()
if timeout_value:
    context_policy['startup_timeout_seconds'] = float(timeout_value)

resume_pass_mode = os.environ.get('BATCH_RESUME_PASS_MODE', '').strip()
if resume_pass_mode:
    context_policy['resume_pass_mode'] = resume_pass_mode

retry_timeout = os.environ.get('BATCH_RETRY_TIMEOUT_FAILURES_ON_RESUME', '').strip().lower()
if retry_timeout:
    context_policy['retry_timeout_failures_on_resume'] = retry_timeout in {'1', 'true', 'yes', 'on'}

retry_oom = os.environ.get('BATCH_RETRY_OOM_FAILURES_ON_RESUME', '').strip().lower()
if retry_oom:
    context_policy['retry_oom_failures_on_resume'] = retry_oom in {'1', 'true', 'yes', 'on'}

retry_infra = os.environ.get('BATCH_RETRY_INFRASTRUCTURE_FAILURES_ON_RESUME', '').strip().lower()
if retry_infra:
    context_policy['retry_infrastructure_failures_on_resume'] = retry_infra in {'1', 'true', 'yes', 'on'}

retry_structural = os.environ.get('BATCH_RETRY_STRUCTURAL_FAILURES_ON_RESUME', '').strip().lower()
if retry_structural:
    context_policy['retry_structural_failures_on_resume'] = retry_structural in {'1', 'true', 'yes', 'on'}

worker_process_mode = os.environ.get('BATCH_WORKER_PROCESS_MODE', '').strip()
if worker_process_mode:
    context_policy['worker_process_mode'] = worker_process_mode

max_requests = os.environ.get('BATCH_MAX_REQUESTS_PER_WORKER_PROCESS', '').strip()
if max_requests:
    context_policy['max_requests_per_worker_process'] = int(float(max_requests))

excluded_labels = {
    label.strip()
    for label in os.environ.get('BATCH_EXCLUDED_DECODING_LABELS', '').split(',')
    if label.strip()
}
exclude_decoding_conditions_from_payload(
    payload,
    excluded_condition_labels=excluded_labels,
)

payload['runtime']['context_policy'] = context_policy
target_path.parent.mkdir(parents=True, exist_ok=True)
target_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
PY

.venv/bin/lcm doctor --json --results-root "$REMOTE_RESULTS_DIR"
.venv/bin/lcm run validate-config --config "$REMOTE_EFFECTIVE_CONFIG_PATH" --output-dir "$REMOTE_PREVIEW_DIR"
