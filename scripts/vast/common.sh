#!/usr/bin/env bash

vast_has_value() {
  local value="$1"
  [ -n "$value" ]
}

vast_ssh_command() {
  local ssh_port="$1"
  local ssh_key_path="$2"
  printf 'ssh -i %q -p %q' "$ssh_key_path" "$ssh_port"
}

vast_rsync_ssh_command() {
  local ssh_port="$1"
  local ssh_key_path="$2"
  printf 'ssh -i %s -p %s' "$ssh_key_path" "$ssh_port"
}

vast_require_positive_integer() {
  local value="$1"
  local variable_name="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || [ "$value" -le 0 ]; then
    echo "$variable_name must be a positive integer" >&2
    return 1
  fi
}
