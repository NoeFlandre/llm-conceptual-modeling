from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, cast

from llm_conceptual_modeling.hf_batch.run_artifacts import (
    clear_retry_artifacts as _clear_retry_artifacts,
)
from llm_conceptual_modeling.hf_batch_types import HFRunSpec, RuntimeResult
from llm_conceptual_modeling.hf_batch_utils import slugify_model as _slugify_model
from llm_conceptual_modeling.hf_batch_utils import write_json as _write_json
from llm_conceptual_modeling.hf_persistent_worker import PersistentHFWorkerSession
from llm_conceptual_modeling.hf_spec_codec import serialize_spec
from llm_conceptual_modeling.hf_subprocess import (
    build_hf_download_environment,
    run_monitored_command,
)


def run_local_hf_spec_subprocess(
    *,
    spec: HFRunSpec,
    run_dir: Path,
    clear_retry_artifacts_fn: Callable[[Path], None] = _clear_retry_artifacts,
    write_json_fn: Callable[[Path, dict[str, Any]], None] = _write_json,
    serialize_spec_fn: Callable[[HFRunSpec], dict[str, object]] = serialize_spec,
    run_monitored_command_fn: Callable[..., subprocess.CompletedProcess[str]] = (
        run_monitored_command
    ),
    build_worker_command_fn: Callable[..., list[str]] | None = None,
    build_hf_download_environment_fn: Callable[[], dict[str, str]] = build_hf_download_environment,
    is_retryable_worker_error_fn: Callable[[dict[str, str]], bool] | None = None,
) -> RuntimeResult:
    startup_timeout_seconds = resolve_startup_timeout_seconds(spec.context_policy)
    stage_timeout_seconds = resolve_stage_timeout_seconds(spec.context_policy)
    retry_attempts = resolve_run_retry_attempts(spec.context_policy)
    spec_json_path = run_dir / "worker_spec.json"
    result_json_path = run_dir / "worker_result.json"
    if build_worker_command_fn is None:
        build_worker_command_fn = build_worker_command
    if is_retryable_worker_error_fn is None:
        is_retryable_worker_error_fn = is_retryable_worker_error

    for attempt in range(1, retry_attempts + 1):
        clear_retry_artifacts_fn(run_dir)
        write_json_fn(spec_json_path, serialize_spec_fn(spec))
        completed = run_monitored_command_fn(
            command=build_worker_command_fn(
                spec_json_path=spec_json_path,
                result_json_path=result_json_path,
                run_dir=run_dir,
            ),
            run_dir=run_dir,
            startup_timeout_seconds=startup_timeout_seconds,
            stage_timeout_seconds=stage_timeout_seconds,
            env=build_hf_download_environment_fn(),
        )
        if not result_json_path.exists():
            raise RuntimeError(
                "HF worker subprocess exited without writing a result artifact. "
                f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
            )
        worker_payload = json.loads(result_json_path.read_text(encoding="utf-8"))
        if worker_payload.get("ok"):
            return cast(RuntimeResult, worker_payload["runtime_result"])

        error = cast(dict[str, str], worker_payload["error"])
        if attempt < retry_attempts and is_retryable_worker_error_fn(error):
            continue
        raise RuntimeError(f"{error['type']}: {error['message']}")

    raise RuntimeError("HF worker retry loop exhausted without a terminal result.")


def run_local_hf_spec(
    *,
    spec: HFRunSpec,
    run_dir: Path,
    output_root: Path,
    persistent_sessions: dict[str, PersistentHFWorkerSession],
) -> RuntimeResult:
    if resolve_worker_process_mode(spec.context_policy) != "persistent":
        return run_local_hf_spec_subprocess(spec=spec, run_dir=run_dir)
    session = persistent_sessions.get(spec.model)
    if session is None:
        queue_dir = output_root / "worker-queues" / _slugify_model(spec.model)
        session = PersistentHFWorkerSession(
            queue_dir=queue_dir,
            worker_python=sys.executable,
            max_requests_per_process=resolve_max_requests_per_worker_process(
                spec.context_policy
            ),
        )
        persistent_sessions[spec.model] = session
    return session.run_spec(spec=spec, run_dir=run_dir)


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


def resolve_startup_timeout_seconds(context_policy: dict[str, object] | None) -> float:
    if context_policy is None:
        return 900.0
    raw_value = context_policy.get("startup_timeout_seconds", 900)
    return coerce_timeout_seconds(raw_value)


def resolve_stage_timeout_seconds(context_policy: dict[str, object] | None) -> float:
    if context_policy is None:
        return 180.0
    raw_value = context_policy.get("generation_timeout_seconds", 180)
    return coerce_timeout_seconds(raw_value)


def resolve_run_retry_attempts(context_policy: dict[str, object] | None) -> int:
    if context_policy is None:
        return 2
    raw_value = context_policy.get("run_retry_attempts", 2)
    if isinstance(raw_value, bool):
        raise TypeError("Retry attempts value must be numeric, got bool")
    if isinstance(raw_value, int | float | str):
        attempts = int(float(raw_value))
        return max(1, attempts)
    raise TypeError(f"Retry attempts value must be numeric, got {type(raw_value).__name__}")


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
            "max_requests_per_worker_process value must be numeric, "
            f"got {type(raw_value).__name__}"
        )
    max_requests = int(float(raw_value))
    if max_requests <= 0:
        raise ValueError("max_requests_per_worker_process must be positive.")
    return max_requests


def is_retryable_worker_error(error: dict[str, str]) -> bool:
    if error.get("type") != "ValueError":
        return False
    message = error.get("message", "")
    retry_markers = (
        "Model did not return valid structured output:",
        "Invalid edge item shape:",
        "Unsupported structured response shape for schema",
        "Structured edge_list response must contain a list of edges",
        "Structured edge_list flat string response must contain an even number of items",
        "Structurally invalid algo",
    )
    return any(marker in message for marker in retry_markers)


def coerce_timeout_seconds(raw_value: object) -> float:
    if isinstance(raw_value, int | float | str):
        return float(raw_value)
    raise TypeError(f"Timeout value must be numeric, got {type(raw_value).__name__}")
