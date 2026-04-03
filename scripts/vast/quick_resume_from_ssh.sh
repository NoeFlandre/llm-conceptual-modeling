#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -ne 4 ]; then
  cat >&2 <<'USAGE'
usage: quick_resume_from_ssh.sh SSH_COMMAND CONFIG_RELATIVE_PATH REMOTE_RESULTS_DIR LOCAL_RESULTS_DIR

Example:
  scripts/vast/quick_resume_from_ssh.sh \
    "ssh -p 35895 root@1.193.139.231 -L 8080:localhost:8080" \
    configs/hf_transformers_algo1_olmo.yaml \
    /workspace/results/hf-paper-batch-algo1-olmo-current \
    /Users/noeflandre/variability-conceptual-modeling/llm-conceptual-modeling/results/hf-paper-batch-algo1-olmo-current
USAGE
  exit 1
fi

SSH_COMMAND="$1"
CONFIG_RELATIVE_PATH="$2"
REMOTE_RESULTS_DIR="$3"
LOCAL_RESULTS_DIR="$4"
LOCAL_REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-/workspace/llm-conceptual-modeling}"

PARSED_SSH="$(
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
)"

SSH_TARGET="$(printf '%s\n' "$PARSED_SSH" | sed -n '1p')"
SSH_PORT="$(printf '%s\n' "$PARSED_SSH" | sed -n '2p')"

exec "$SCRIPT_DIR/prepare_and_resume_hf_batch.sh" \
  "$SSH_TARGET" \
  "$SSH_PORT" \
  "$LOCAL_REPO_DIR" \
  "$REMOTE_REPO_DIR" \
  "$CONFIG_RELATIVE_PATH" \
  "$REMOTE_RESULTS_DIR" \
  "$LOCAL_RESULTS_DIR"
