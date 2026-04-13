from llm_conceptual_modeling.hf_execution import (
    build_worker_command,
    coerce_timeout_seconds,
    is_retryable_worker_error,
    resolve_max_requests_per_worker_process,
    resolve_worker_process_mode,
    run_local_hf_spec_subprocess,
)
from llm_conceptual_modeling.hf_execution import runtime as _execution_runtime
from llm_conceptual_modeling.hf_persistent_worker import PersistentHFWorkerSession
from llm_conceptual_modeling.hf_worker_policy import (
    resolve_run_retry_attempts,
    resolve_stage_timeout_seconds,
    resolve_startup_timeout_seconds,
)

__all__ = [
    "build_worker_command",
    "coerce_timeout_seconds",
    "is_retryable_worker_error",
    "resolve_max_requests_per_worker_process",
    "resolve_run_retry_attempts",
    "resolve_stage_timeout_seconds",
    "resolve_startup_timeout_seconds",
    "resolve_worker_process_mode",
    "PersistentHFWorkerSession",
    "run_local_hf_spec",
    "run_local_hf_spec_subprocess",
]


def _close_incompatible_persistent_sessions(*, spec, persistent_sessions) -> None:
    from llm_conceptual_modeling.hf_execution.runtime import (
        _close_incompatible_persistent_sessions as _close_sessions,
    )

    _close_sessions(spec=spec, persistent_sessions=persistent_sessions)


def run_local_hf_spec(*, spec, run_dir, output_root, persistent_sessions):
    return _execution_runtime.run_local_hf_spec(
        spec=spec,
        run_dir=run_dir,
        output_root=output_root,
        persistent_sessions=persistent_sessions,
        persistent_worker_session_cls=PersistentHFWorkerSession,
    )
