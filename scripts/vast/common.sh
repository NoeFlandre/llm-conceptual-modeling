#!/usr/bin/env bash

VAST_SSH_CONNECT_TIMEOUT_SECONDS="${VAST_SSH_CONNECT_TIMEOUT_SECONDS:-60}"
VAST_SSH_SERVER_ALIVE_INTERVAL_SECONDS="${VAST_SSH_SERVER_ALIVE_INTERVAL_SECONDS:-30}"
VAST_SSH_SERVER_ALIVE_COUNT_MAX="${VAST_SSH_SERVER_ALIVE_COUNT_MAX:-6}"
VAST_SSH_CONTROL_PATH="${VAST_SSH_CONTROL_PATH:-$HOME/.ssh/lcm-vast-%C}"
VAST_SSH_CONTROL_PERSIST_SECONDS="${VAST_SSH_CONTROL_PERSIST_SECONDS:-600}"

vast_has_value() {
  local value="$1"
  [ -n "$value" ]
}

vast_parse_ssh_command() {
  local ssh_command="$1"
  SSH_COMMAND="$ssh_command" python3 - <<'PY'
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

vast_ssh_transport_flags() {
  printf '%s' \
    "-o ConnectTimeout=${VAST_SSH_CONNECT_TIMEOUT_SECONDS} " \
    "-o ServerAliveInterval=${VAST_SSH_SERVER_ALIVE_INTERVAL_SECONDS} " \
    "-o ServerAliveCountMax=${VAST_SSH_SERVER_ALIVE_COUNT_MAX} " \
    "-o ControlMaster=auto " \
    "-o ControlPersist=${VAST_SSH_CONTROL_PERSIST_SECONDS} " \
    "-o ControlPath=${VAST_SSH_CONTROL_PATH} " \
    "-o StrictHostKeyChecking=accept-new"
}

vast_ssh_command() {
  local ssh_port="$1"
  local ssh_key_path="$2"
  printf 'ssh %s -i %q -p %q' "$(vast_ssh_transport_flags)" "$ssh_key_path" "$ssh_port"
}

vast_rsync_ssh_command() {
  local ssh_port="$1"
  local ssh_key_path="$2"
  printf 'ssh %s -i %s -p %s' "$(vast_ssh_transport_flags)" "$ssh_key_path" "$ssh_port"
}

vast_rsync_resume_flags() {
  local timeout_seconds="$1"
  printf '%s' "--partial --timeout ${timeout_seconds}"
}

vast_retry_rsync() {
  local attempts="$1"
  shift
  local attempt=1
  local status=0

  while true; do
    if "$@"; then
      return 0
    fi
    status=$?
    if [ "$attempt" -ge "$attempts" ]; then
      return "$status"
    fi
    sleep 5
    attempt=$((attempt + 1))
  done
}

vast_require_positive_integer() {
  local value="$1"
  local variable_name="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || [ "$value" -le 0 ]; then
    echo "$variable_name must be a positive integer" >&2
    return 1
  fi
}

vast_select_remote_runtime_mode() {
  local runtime_mode="$1"
  local docker_image="$2"
  if [ "$runtime_mode" = "auto" ]; then
    if vast_has_value "$docker_image"; then
      printf '%s\n' "docker"
    else
      printf '%s\n' "bootstrap"
    fi
    return 0
  fi
  printf '%s\n' "$runtime_mode"
}

vast_watcher_identity() {
  local ssh_target="$1"
  local ssh_port="$2"
  local remote_results_dir="$3"
  printf '%s\n' "${ssh_target}:${ssh_port}:${remote_results_dir}"
}
