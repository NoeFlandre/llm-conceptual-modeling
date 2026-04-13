from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from llm_conceptual_modeling.common.spec_codec import serialize_spec
from llm_conceptual_modeling.hf_batch.run_artifacts import (
    clear_retry_artifacts as _clear_retry_artifacts,
)
from llm_conceptual_modeling.hf_batch.types import HFRunSpec, RuntimeResult
from llm_conceptual_modeling.hf_batch.utils import slugify_model as _slugify_model
from llm_conceptual_modeling.hf_batch.utils import write_json as _write_json
from llm_conceptual_modeling.hf_execution.helpers import (
    _is_retryable_missing_result_artifact,
    build_worker_command,
    is_retryable_worker_error,
    resolve_max_requests_per_worker_process,
    resolve_worker_process_mode,
)
from llm_conceptual_modeling.hf_execution.subprocess import (
    MonitoredCommandTimeout,
    build_hf_download_environment,
    run_monitored_command,
)
from llm_conceptual_modeling.hf_worker.persistent import PersistentHFWorkerSession
from llm_conceptual_modeling.hf_worker_policy import (
    resolve_run_retry_attempts,
    resolve_stage_timeout_seconds,
    resolve_startup_timeout_seconds,
)
from llm_conceptual_modeling.hf_worker_result import (
    load_runtime_result,
    raise_missing_result_artifact,
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
    validate_runtime_result_fn: Callable[[str, dict[str, object]], None] | None = None,
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
        try:
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
        except MonitoredCommandTimeout:
            if attempt < retry_attempts:
                continue
            raise
        if not result_json_path.exists():
            if attempt < retry_attempts and _is_retryable_missing_result_artifact(
                stdout=completed.stdout,
                stderr=completed.stderr,
            ):
                continue
            raise_missing_result_artifact(
                context="HF worker subprocess",
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        try:
            runtime_result = load_runtime_result(result_json_path)
        except RuntimeError as error:
            if attempt >= retry_attempts:
                raise
            message = str(error)
            error_type, _separator, error_message = message.partition(": ")
            error_payload = {"type": error_type, "message": error_message}
            if is_retryable_worker_error_fn(error_payload):
                continue
            raise
        if validate_runtime_result_fn is None:
            return runtime_result
        try:
            validate_runtime_result_fn(
                algorithm=spec.algorithm,
                raw_row=runtime_result["raw_row"],
            )
        except Exception as error:
            if attempt >= retry_attempts:
                raise
            error_payload = {"type": type(error).__name__, "message": str(error)}
            if is_retryable_worker_error_fn(error_payload):
                continue
            raise
        return runtime_result

    raise RuntimeError("HF worker retry loop exhausted without a terminal result.")


def run_local_hf_spec(
    *,
    spec: HFRunSpec,
    run_dir: Path,
    output_root: Path,
    persistent_sessions: dict[str, PersistentHFWorkerSession],
    persistent_worker_session_cls: type[PersistentHFWorkerSession] = PersistentHFWorkerSession,
) -> RuntimeResult:
    _close_incompatible_persistent_sessions(
        spec=spec,
        persistent_sessions=persistent_sessions,
    )
    if resolve_worker_process_mode(spec.context_policy) != "persistent":
        return run_local_hf_spec_subprocess(spec=spec, run_dir=run_dir)
    session = persistent_sessions.get(spec.model)
    if session is None:
        queue_dir = output_root / "worker-queues" / _slugify_model(spec.model)
        session = persistent_worker_session_cls(
            queue_dir=queue_dir,
            worker_python=sys.executable,
            max_requests_per_process=resolve_max_requests_per_worker_process(spec.context_policy),
        )
        persistent_sessions[spec.model] = session
    return session.run_spec(spec=spec, run_dir=run_dir)


def _close_incompatible_persistent_sessions(
    *,
    spec: HFRunSpec,
    persistent_sessions: dict[str, PersistentHFWorkerSession],
) -> None:
    if not persistent_sessions:
        return
    model_names_to_close = [
        model_name
        for model_name in persistent_sessions
        if model_name != spec.model
    ]
    for model_name in model_names_to_close:
        session = persistent_sessions.pop(model_name)
        session.close()
