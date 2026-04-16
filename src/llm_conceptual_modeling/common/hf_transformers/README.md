# `common/hf_transformers`

Packaged Hugging Face Transformers helpers live here.

## Contents

- `_policy.py` - decoding policy, capability checks, and context helpers
- `_parse.py` - structured-output parsing and recovery orchestration
- `_qwen.py` - Qwen contrastive-cache compatibility helpers
- `_runtime.py` - runtime loading, hardware resolution, and factory assembly
- `_client.py` - chat and embedding client classes plus generation helpers
- `__init__.py` - compatibility facade that re-exports the public surface
- `_compat.py` - thin backward-compatibility shim for older import paths

## Maintenance Rule

Keep this package narrowly focused on HF Transformers runtime integration.
If a helper is reusable outside the runtime boundary, move it only when the
new home is explicit and stable.
