#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

if [ "$#" -ne 1 ]; then
  echo "usage: remote_runtime_doctor.sh REMOTE_REPO_DIR" >&2
  exit 1
fi

REMOTE_REPO_DIR="$1"
REMOTE_RUNTIME_MODE="${REMOTE_RUNTIME_MODE:-bootstrap}"
REMOTE_DOCKER_IMAGE="${REMOTE_DOCKER_IMAGE:-}"

cd "$REMOTE_REPO_DIR"
export PYTHONPATH="$REMOTE_REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

if [ "$REMOTE_RUNTIME_MODE" = "docker" ]; then
  command -v docker >/dev/null 2>&1
  if [ -z "$REMOTE_DOCKER_IMAGE" ]; then
    echo "REMOTE_DOCKER_IMAGE must be set for docker runtime doctor" >&2
    exit 1
  fi
  docker image inspect "$REMOTE_DOCKER_IMAGE" >/dev/null 2>&1
  docker run --rm "$REMOTE_DOCKER_IMAGE" bash -lc 'test -x .venv/bin/lcm'
  exit 0
fi

if [ "$REMOTE_RUNTIME_MODE" = "bootstrap" ]; then
  command -v bash >/dev/null 2>&1
  command -v curl >/dev/null 2>&1
  command -v nvidia-smi >/dev/null 2>&1
  exit 0
fi

echo "Unsupported REMOTE_RUNTIME_MODE: $REMOTE_RUNTIME_MODE" >&2
exit 1
