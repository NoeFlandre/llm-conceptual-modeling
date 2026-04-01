#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-$HOME/llm-conceptual-modeling}"
export TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
export TORCH_VERSION="${TORCH_VERSION:-2.5.1+cu121}"
export TRITON_VERSION="${TRITON_VERSION:-3.1.0}"
export TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.4.0}"
BOOTSTRAP_SNAPSHOT_PATH="${BOOTSTRAP_SNAPSHOT_PATH:-$REPO_DIR/.bootstrap-runtime.json}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

cd "$REPO_DIR"
if [ -x .venv/bin/python ]; then
  TORCH_HEALTH_OUTPUT="$(
    { .venv/bin/python - <<'PY' 2>&1; } || true
try:
    import torch
    print("ok")
except Exception as exc:
    print(repr(exc))
PY
  )"
  if printf '%s' "$TORCH_HEALTH_OUTPUT" | grep -q "undefined symbol: ncclCommWindowDeregister"; then
    rm -rf .venv
  fi

  if .venv/bin/python - <<'PY' >"$BOOTSTRAP_SNAPSHOT_PATH" 2>/dev/null; then
import json
import os
import sys
from datetime import datetime, timezone

import torch
import transformers
import triton

if not torch.cuda.is_available():
    raise SystemExit(1)
if torch.__version__ != os.environ["TORCH_VERSION"]:
    raise SystemExit(1)
if triton.__version__ != os.environ["TRITON_VERSION"]:
    raise SystemExit(1)
if transformers.__version__ != os.environ["TRANSFORMERS_VERSION"]:
    raise SystemExit(1)

print(json.dumps({
    "python_version": sys.version.split()[0],
    "cuda_device": torch.cuda.get_device_name(0),
    "bf16": torch.cuda.is_bf16_supported(),
    "torch_version": torch.__version__,
    "cuda_version": getattr(torch.version, "cuda", None),
    "transformers_version": transformers.__version__,
    "triton_version": triton.__version__,
    "timestamp": datetime.now(timezone.utc).isoformat(),
}, indent=2))
PY
    cat "$BOOTSTRAP_SNAPSHOT_PATH"
    exit 0
  fi
fi

uv sync --no-install-package torch --no-install-package triton
uv pip install --python .venv/bin/python wheel
uv pip install \
  --python .venv/bin/python \
  --index-url "$TORCH_INDEX_URL" \
  --upgrade \
  "torch==$TORCH_VERSION"
uv pip install \
  --python .venv/bin/python \
  --no-deps \
  --upgrade \
  "triton==$TRITON_VERSION"
.venv/bin/python - <<'PY' >"$BOOTSTRAP_SNAPSHOT_PATH"
import json
import sys
from datetime import datetime, timezone

import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; aborting bootstrap.")
import transformers
try:
    import triton
    triton_version = triton.__version__
except Exception:
    triton_version = None
print(json.dumps({
    "python_version": sys.version.split()[0],
    "cuda_device": torch.cuda.get_device_name(0),
    "bf16": torch.cuda.is_bf16_supported(),
    "torch_version": torch.__version__,
    "cuda_version": getattr(torch.version, "cuda", None),
    "transformers_version": transformers.__version__,
    "triton_version": triton_version,
    "timestamp": datetime.now(timezone.utc).isoformat(),
}, indent=2))
PY
cat "$BOOTSTRAP_SNAPSHOT_PATH"
