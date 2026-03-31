#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-$HOME/llm-conceptual-modeling}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

cd "$REPO_DIR"
uv sync
uv pip install --python .venv/bin/python --index-url "$TORCH_INDEX_URL" --upgrade torch
uv run python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; aborting bootstrap.")
print({"cuda_device": torch.cuda.get_device_name(0), "bf16": torch.cuda.is_bf16_supported()})
PY
