from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from llm_conceptual_modeling.hf_batch.types import RuntimeResult


def load_runtime_result(result_json_path: Path) -> RuntimeResult:
    worker_payload = json.loads(result_json_path.read_text(encoding="utf-8"))
    if worker_payload.get("ok"):
        return cast(RuntimeResult, worker_payload["runtime_result"])
    error = cast(dict[str, str], worker_payload["error"])
    raise RuntimeError(f"{error['type']}: {error['message']}")


def raise_missing_result_artifact(
    *,
    context: str,
    stdout: str | None = None,
    stderr: str | None = None,
) -> None:
    raise RuntimeError(
        f"{context} exited without writing a result artifact. "
        f"stdout={stdout!r} stderr={stderr!r}"
    )
