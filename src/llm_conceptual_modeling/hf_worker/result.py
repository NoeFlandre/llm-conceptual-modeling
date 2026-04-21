from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, cast

from llm_conceptual_modeling.hf_batch.types import RuntimeResult


class HFWorkerErrorPayload(TypedDict):
    type: str
    message: str


class HFWorkerSuccessPayload(TypedDict):
    ok: bool
    runtime_result: RuntimeResult


def load_runtime_result(result_json_path: Path) -> RuntimeResult:
    worker_payload = json.loads(result_json_path.read_text(encoding="utf-8"))
    if not isinstance(worker_payload, dict):
        raise RuntimeError("Malformed HF worker payload: expected a JSON object.")
    if worker_payload.get("ok") is True:
        runtime_result = worker_payload.get("runtime_result")
        if not isinstance(runtime_result, dict):
            raise RuntimeError("Malformed HF worker success payload: missing runtime_result.")
        success_payload = cast(
            HFWorkerSuccessPayload,
            {"ok": True, "runtime_result": cast(RuntimeResult, runtime_result)},
        )
        return success_payload["runtime_result"]
    error_payload = _load_worker_error_payload(worker_payload)
    raise RuntimeError(f"{error_payload['type']}: {error_payload['message']}")


def _load_worker_error_payload(worker_payload: dict[str, object]) -> HFWorkerErrorPayload:
    error = worker_payload.get("error")
    if not isinstance(error, dict):
        raise RuntimeError("Malformed HF worker error payload: missing error object.")
    error_payload = cast(dict[str, object], error)
    error_type = error_payload.get("type")
    error_message = error_payload.get("message")
    if not isinstance(error_type, str) or not isinstance(error_message, str):
        raise RuntimeError("Malformed HF worker error payload: missing type/message.")
    return {"type": error_type, "message": error_message}


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
