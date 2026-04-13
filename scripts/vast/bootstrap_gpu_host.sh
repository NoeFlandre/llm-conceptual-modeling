#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-$HOME/llm-conceptual-modeling}"
export TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
export TORCH_VERSION="${TORCH_VERSION:-2.8.0+cu128}"
export TRITON_VERSION="${TRITON_VERSION:-3.6.0}"
export TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.4.0}"
BOOTSTRAP_SNAPSHOT_PATH="${BOOTSTRAP_SNAPSHOT_PATH:-$REPO_DIR/.bootstrap-runtime.json}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-0}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

vast_retry() {
  local attempts="$1"
  local delay_seconds="$2"
  shift 2

  local attempt=1
  while true; do
    if "$@"; then
      return 0
    fi

    if [ "$attempt" -ge "$attempts" ]; then
      return 1
    fi

    sleep "$delay_seconds"
    attempt=$((attempt + 1))
  done
}

vast_torch_runtime_probe() {
  .venv/bin/python - <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

import torch
import transformers
import triton

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; aborting bootstrap.")
if torch.__version__ != os.environ["TORCH_VERSION"]:
    raise SystemExit(1)
if triton.__version__ != os.environ["TRITON_VERSION"]:
    raise SystemExit(1)
if transformers.__version__ != os.environ["TRANSFORMERS_VERSION"]:
    raise SystemExit(1)

probe = torch.ones(1, device="cuda")
probe = probe + 1
torch.cuda.synchronize()

print(json.dumps({
    "python_version": sys.version.split()[0],
    "cuda_device": torch.cuda.get_device_name(0),
    "cuda_capability": list(torch.cuda.get_device_capability(0)),
    "bf16": torch.cuda.is_bf16_supported(),
    "cuda_kernel_smoke": True,
    "torch_version": torch.__version__,
    "cuda_version": getattr(torch.version, "cuda", None),
    "transformers_version": transformers.__version__,
    "triton_version": triton.__version__,
    "timestamp": datetime.now(timezone.utc).isoformat(),
}, indent=2))
PY
}

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

cd "$REPO_DIR"
if [ -x .venv/bin/python ]; then
  if TORCH_HEALTH_OUTPUT="$(vast_torch_runtime_probe 2>&1)"; then
    printf '%s\n' "$TORCH_HEALTH_OUTPUT" >"$BOOTSTRAP_SNAPSHOT_PATH"
    cat "$BOOTSTRAP_SNAPSHOT_PATH"
    exit 0
  fi
  if printf '%s' "$TORCH_HEALTH_OUTPUT" | grep -q "undefined symbol: ncclCommWindowDeregister"; then
    rm -rf .venv
  elif printf '%s' "$TORCH_HEALTH_OUTPUT" | grep -Eq 'CUDA is not available; aborting bootstrap\.|no kernel image is available for execution on the device'; then
    rm -rf .venv
  else
    rm -rf .venv
  fi
fi

vast_retry 3 10 uv sync --no-install-package torch --no-install-package triton
vast_retry 3 10 uv pip install --python .venv/bin/python wheel
vast_retry 3 10 uv pip install \
  --python .venv/bin/python \
  --index-url "$TORCH_INDEX_URL" \
  --upgrade \
  "torch==$TORCH_VERSION"
vast_retry 3 10 uv pip install \
  --python .venv/bin/python \
  --no-deps \
  --upgrade \
  "triton==$TRITON_VERSION"

if TORCH_HEALTH_OUTPUT="$(vast_torch_runtime_probe 2>&1)"; then
  printf '%s\n' "$TORCH_HEALTH_OUTPUT" >"$BOOTSTRAP_SNAPSHOT_PATH"
  cat "$BOOTSTRAP_SNAPSHOT_PATH"
else
  printf '%s\n' "$TORCH_HEALTH_OUTPUT" >&2
  exit 1
fi
