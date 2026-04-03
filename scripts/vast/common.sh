#!/usr/bin/env bash

VAST_SSH_CONNECT_TIMEOUT_SECONDS="${VAST_SSH_CONNECT_TIMEOUT_SECONDS:-10}"
VAST_SSH_SERVER_ALIVE_INTERVAL_SECONDS="${VAST_SSH_SERVER_ALIVE_INTERVAL_SECONDS:-30}"
VAST_SSH_SERVER_ALIVE_COUNT_MAX="${VAST_SSH_SERVER_ALIVE_COUNT_MAX:-6}"

vast_has_value() {
  local value="$1"
  [ -n "$value" ]
}

vast_ssh_transport_flags() {
  printf '%s' \
    "-o ConnectTimeout=${VAST_SSH_CONNECT_TIMEOUT_SECONDS} " \
    "-o ServerAliveInterval=${VAST_SSH_SERVER_ALIVE_INTERVAL_SECONDS} " \
    "-o ServerAliveCountMax=${VAST_SSH_SERVER_ALIVE_COUNT_MAX}"
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
