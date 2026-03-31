#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-$HOME/llm-conceptual-modeling}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1+cu121}"
TRITON_VERSION="${TRITON_VERSION:-3.1.0}"
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
  --upgrade \
  "triton==$TRITON_VERSION"
uv run python - <<'PY'
import json
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
    "cuda_device": torch.cuda.get_device_name(0),
    "bf16": torch.cuda.is_bf16_supported(),
    "torch_version": torch.__version__,
    "cuda_version": getattr(torch.version, "cuda", None),
    "transformers_version": transformers.__version__,
    "triton_version": triton_version,
}, indent=2))
PY
