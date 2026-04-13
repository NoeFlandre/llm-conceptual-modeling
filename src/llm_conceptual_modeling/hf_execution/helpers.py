from __future__ import annotations

import sys
from pathlib import Path

from llm_conceptual_modeling.common.failure_markers import is_retryable_runtime_failure
from llm_conceptual_modeling.hf_worker_policy import (
    coerce_timeout_seconds as _worker_policy_coerce_timeout_seconds,
)


def build_worker_command(
    *,
    spec_json_path: Path,
    result_json_path: Path,
    run_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "llm_conceptual_modeling.hf_worker",
        "--spec-json",
        str(spec_json_path),
        "--result-json",
        str(result_json_path),
        "--run-dir",
        str(run_dir),
    ]


def resolve_worker_process_mode(context_policy: dict[str, object] | None) -> str:
    if context_policy is None:
        return "ephemeral"
    raw_value = context_policy.get("worker_process_mode", "ephemeral")
    if not isinstance(raw_value, str):
        raise ValueError(
            "worker_process_mode value must be string, "
            f"received {raw_value!r} ({type(raw_value).__name__})."
        )
    mode = raw_value.strip().lower()
    if mode not in {"ephemeral", "persistent"}:
        raise ValueError(
            "worker_process_mode must be one of {'ephemeral', 'persistent'}, "
            f"received {raw_value!r}."
        )
    return mode


def resolve_max_requests_per_worker_process(
    context_policy: dict[str, object] | None,
) -> int | None:
    if context_policy is None:
        return None
    raw_value = context_policy.get("max_requests_per_worker_process")
    if raw_value is None:
        return None
    if isinstance(raw_value, bool) or not isinstance(raw_value, int | float | str):
        raise TypeError(
            f"max_requests_per_worker_process value must be numeric, got {type(raw_value).__name__}"
        )
    max_requests = int(float(raw_value))
    if max_requests <= 0:
        raise ValueError("max_requests_per_worker_process must be positive.")
    return max_requests


def is_retryable_worker_error(error: dict[str, str]) -> bool:
    return is_retryable_runtime_failure(
        error_type=error.get("type", ""),
        message=error.get("message", ""),
    )


def coerce_timeout_seconds(raw_value: object) -> float:
    return _worker_policy_coerce_timeout_seconds(raw_value)


def _is_retryable_missing_result_artifact(*, stdout: str | None, stderr: str | None) -> bool:
    combined_output = "\n".join(part for part in (stdout or "", stderr or "") if part)
    return is_retryable_runtime_failure(
        error_type="RuntimeError",
        message=combined_output,
    )
