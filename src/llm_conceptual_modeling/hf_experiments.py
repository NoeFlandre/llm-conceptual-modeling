from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from llm_conceptual_modeling.common.hf_transformers import (
    build_runtime_factory,
)
from llm_conceptual_modeling.hf_batch.monitoring import (
    current_run_payload as _current_run_payload,
)
from llm_conceptual_modeling.hf_batch.monitoring import (
    status_timestamp_now as _status_timestamp_now,
)
from llm_conceptual_modeling.hf_batch.monitoring import (
    write_status_snapshot as _write_status_snapshot,
)
from llm_conceptual_modeling.hf_batch.outputs import write_aggregated_outputs
from llm_conceptual_modeling.hf_batch.planning import (
    default_runtime_profile_provider,
    plan_paper_batch,  # noqa: F401
    select_run_spec,  # noqa: F401
)
from llm_conceptual_modeling.hf_batch.planning import (
    plan_paper_batch_specs as _plan_paper_batch_specs,
)
from llm_conceptual_modeling.hf_batch.prompts import (
    build_prompt_bundle as _build_prompt_bundle,  # noqa: F401
)
from llm_conceptual_modeling.hf_batch.run_artifacts import (
    build_run_summary as _build_run_summary,
)
from llm_conceptual_modeling.hf_batch.run_artifacts import (
    clear_retry_artifacts as _clear_retry_artifacts,
)
from llm_conceptual_modeling.hf_batch.run_artifacts import (
    normalize_stale_running_run as _normalize_stale_running_run,
)
from llm_conceptual_modeling.hf_batch.run_artifacts import (
    read_json as _read_artifact_json,
)
from llm_conceptual_modeling.hf_batch.run_artifacts import (
    write_run_artifacts as _write_run_artifacts,
)
from llm_conceptual_modeling.hf_batch.run_artifacts import (
    write_smoke_verdict as _write_smoke_verdict,
)
from llm_conceptual_modeling.hf_batch.spec_path import (
    filter_planned_specs_for_output_root as _filter_planned_specs_for_output_root,
)
from llm_conceptual_modeling.hf_batch.spec_path import (
    run_dir_for_spec as _run_dir_for_spec,
)
from llm_conceptual_modeling.hf_batch.spec_path import (
    smoke_spec_identity as _smoke_spec_identity,
)
from llm_conceptual_modeling.hf_batch.types import (
    BatchInfrastructureFailure,
    HFRunSpec,
    RuntimeFactory,
    RuntimeResult,
)
from llm_conceptual_modeling.hf_batch.utils import (
    manifest_for_spec as _manifest_for_spec,
)
from llm_conceptual_modeling.hf_batch.utils import (
    resolve_hf_token as _resolve_hf_token,
)
from llm_conceptual_modeling.hf_batch.utils import (
    slugify_model as _slugify_model,  # noqa: F401
)
from llm_conceptual_modeling.hf_batch.utils import (
    write_json as _write_json,
)
from llm_conceptual_modeling.hf_execution.dispatch import (
    execute_run as _execute_run,
)
from llm_conceptual_modeling.hf_execution.dispatch import (
    runtime_factory_from_hf_runtime as _runtime_factory_from_hf_runtime,
)
from llm_conceptual_modeling.hf_execution.helpers import (
    build_worker_command as _build_worker_command,  # noqa: F401
)
from llm_conceptual_modeling.hf_execution.helpers import (
    coerce_timeout_seconds as _coerce_timeout_seconds,  # noqa: F401
)
from llm_conceptual_modeling.hf_execution.helpers import (
    is_retryable_worker_error as _is_retryable_worker_error,  # noqa: F401
)
from llm_conceptual_modeling.hf_execution.helpers import (
    resolve_max_requests_per_worker_process as _resolve_max_requests_per_worker_process,
)
from llm_conceptual_modeling.hf_execution.helpers import (
    resolve_run_retry_attempts as _resolve_run_retry_attempts,  # noqa: F401
)
from llm_conceptual_modeling.hf_execution.helpers import (
    resolve_stage_timeout_seconds as _resolve_stage_timeout_seconds,  # noqa: F401
)
from llm_conceptual_modeling.hf_execution.helpers import (
    resolve_startup_timeout_seconds as _resolve_startup_timeout_seconds,  # noqa: F401
)
from llm_conceptual_modeling.hf_execution.helpers import (
    resolve_worker_process_mode as _resolve_worker_process_mode,  # noqa: F401
)
from llm_conceptual_modeling.hf_execution.runtime import (
    run_local_hf_spec_subprocess as _run_local_hf_spec_subprocess_impl,
)
from llm_conceptual_modeling.hf_execution.subprocess import run_monitored_command  # noqa: F401
from llm_conceptual_modeling.hf_pipeline.algo1 import run_algo1 as _run_algo1  # noqa: F401
from llm_conceptual_modeling.hf_pipeline.algo2 import run_algo2 as _run_algo2  # noqa: F401
from llm_conceptual_modeling.hf_pipeline.algo3 import run_algo3 as _run_algo3  # noqa: F401
from llm_conceptual_modeling.hf_pipeline.metrics import (
    connection_metric_summary as _connection_metric_summary,  # noqa: F401
)
from llm_conceptual_modeling.hf_pipeline.metrics import (
    sanitize_algorithm_edge_result as _sanitize_algorithm_edge_result,  # noqa: F401
)
from llm_conceptual_modeling.hf_pipeline.metrics import (
    summary_from_raw_row as _summary_from_raw_row,  # noqa: F401
)
from llm_conceptual_modeling.hf_pipeline.metrics import (
    trace_metric_summary as _trace_metric_summary,  # noqa: F401
)
from llm_conceptual_modeling.hf_pipeline.metrics import (
    validate_structural_runtime_result as _validate_structural_runtime_result,
)
from llm_conceptual_modeling.hf_state.resume_state import (
    build_seeded_resume_snapshot as _resume_build_seeded_resume_snapshot,
)
from llm_conceptual_modeling.hf_state.resume_state import (
    classify_failure_payload as _classify_failure_payload,
)
from llm_conceptual_modeling.hf_state.resume_state import (
    collect_resume_history as _collect_resume_history,  # noqa: F401
)
from llm_conceptual_modeling.hf_state.resume_state import (
    is_finished_run_directory as _is_finished_run_directory,  # noqa: F401
)
from llm_conceptual_modeling.hf_state.resume_state import (
    load_deferred_failed_summary as _load_deferred_failed_summary,
)
from llm_conceptual_modeling.hf_state.resume_state import (
    load_valid_finished_summary as _load_valid_finished_summary,
)
from llm_conceptual_modeling.hf_state.resume_state import (
    order_planned_specs_for_resume as _resume_order_planned_specs_for_resume,
)
from llm_conceptual_modeling.hf_state.resume_state import (
    resolve_resume_pass_mode as _resolve_resume_pass_mode,  # noqa: F401
)
from llm_conceptual_modeling.hf_state.resume_state import (
    should_keep_failure_pending_on_resume as _should_keep_failure_pending_on_resume,
)
from llm_conceptual_modeling.hf_state.resume_state import (
    status_failures as _status_failures,
)
from llm_conceptual_modeling.hf_state.resume_state import (
    status_int as _status_int,
)
from llm_conceptual_modeling.hf_worker.persistent import PersistentHFWorkerSession
from llm_conceptual_modeling.hf_worker.state import (
    worker_loaded_model as _worker_loaded_model,  # noqa: F401
)

plan_paper_batch_specs = _plan_paper_batch_specs


def _run_local_hf_spec_subprocess(*, spec: HFRunSpec, run_dir: Path) -> RuntimeResult:
    return _run_local_hf_spec_subprocess_impl(
        spec=spec,
        run_dir=run_dir,
        run_monitored_command_fn=run_monitored_command,
        build_worker_command_fn=_build_worker_command,
        is_retryable_worker_error_fn=_is_retryable_worker_error,
        validate_runtime_result_fn=_validate_structural_runtime_result,
    )


def _run_local_hf_spec(
    *,
    spec: HFRunSpec,
    run_dir: Path,
    output_root: Path,
    persistent_sessions: dict[str, PersistentHFWorkerSession],
) -> RuntimeResult:
    _close_incompatible_persistent_sessions(
        model=spec.model,
        persistent_sessions=persistent_sessions,
    )
    if _resolve_worker_process_mode(spec.context_policy) != "persistent":
        return _run_local_hf_spec_subprocess(spec=spec, run_dir=run_dir)
    session = _get_persistent_session(
        spec=spec,
        output_root=output_root,
        persistent_sessions=persistent_sessions,
    )
    return session.run_spec(spec=spec, run_dir=run_dir)


def _close_incompatible_persistent_sessions(
    *,
    model: str,
    persistent_sessions: dict[str, PersistentHFWorkerSession],
) -> None:
    model_names_to_close = [
        model_name
        for model_name in persistent_sessions
        if model_name != model
    ]
    for model_name in model_names_to_close:
        session = persistent_sessions.pop(model_name)
        session.close()

def _get_persistent_session(
    *,
    spec: HFRunSpec,
    output_root: Path,
    persistent_sessions: dict[str, PersistentHFWorkerSession],
) -> PersistentHFWorkerSession:
    session = persistent_sessions.get(spec.model)
    if session is None:
        queue_dir = output_root / "worker-queues" / _slugify_model(spec.model)
        session = PersistentHFWorkerSession(
            queue_dir=queue_dir,
            worker_python=sys.executable,
            max_requests_per_process=_resolve_max_requests_per_worker_process(
                spec.context_policy
            ),
        )
        persistent_sessions[spec.model] = session
    return session


@dataclass(frozen=True)
class _BatchSetup:
    output_root: Path
    planned_specs: list[HFRunSpec]
    runtime_factory: RuntimeFactory | None
    use_monitored_hf_subprocess: bool
    dry_run: bool
    resume: bool


@dataclass(frozen=True)
class _BatchArguments:
    output_root: str | Path
    models: list[str]
    embedding_model: str
    replications: int


@dataclass(frozen=True)
class _BatchRuntime:
    runtime_factory: RuntimeFactory | None
    profile_provider: Any
    use_monitored_hf_subprocess: bool


@dataclass
class _BatchState:
    total_runs: int
    planned_specs: list[HFRunSpec]
    status_snapshot: dict[str, object]
    summary_rows: list[dict[str, object]]
    persistent_sessions: dict[str, PersistentHFWorkerSession]
    seeded_finished_run_dirs: set[Path]
    seeded_failed_run_dirs: set[Path]


def run_paper_batch(
    *,
    output_root: str | Path,
    models: list[str],
    embedding_model: str,
    replications: int,
    algorithms: tuple[str, ...] | None = None,
    config: Any | None = None,
    runtime_factory: RuntimeFactory | None = None,
    resume: bool = False,
    dry_run: bool = False,
) -> None:
    setup = _prepare_batch_setup(
        output_root=output_root,
        models=models,
        embedding_model=embedding_model,
        replications=replications,
        algorithms=algorithms,
        config=config,
        runtime_factory=runtime_factory,
        resume=resume,
        dry_run=dry_run,
    )
    state = _initialize_batch_state(setup)
    _write_status_snapshot(output_root=setup.output_root, status=state.status_snapshot)

    try:
        for spec in state.planned_specs:
            _process_planned_spec(setup=setup, state=state, spec=spec)
    finally:
        for session in state.persistent_sessions.values():
            session.close()

    _finalize_batch_outputs(setup=setup, state=state)


def _prepare_batch_setup(
    *,
    output_root: str | Path,
    models: list[str],
    embedding_model: str,
    replications: int,
    algorithms: tuple[str, ...] | None,
    config: Any | None,
    runtime_factory: RuntimeFactory | None,
    resume: bool,
    dry_run: bool,
) -> _BatchSetup:
    arguments = _resolve_batch_arguments(
        output_root=output_root,
        models=models,
        embedding_model=embedding_model,
        replications=replications,
        config=config,
    )
    output_root_path = Path(arguments.output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    runtime = _resolve_batch_runtime(runtime_factory=runtime_factory, dry_run=dry_run)
    planned_specs = plan_paper_batch_specs(
        models=arguments.models,
        embedding_model=arguments.embedding_model,
        replications=arguments.replications,
        algorithms=algorithms,
        config=config,
        runtime_profile_provider=runtime.profile_provider,
    )
    planned_specs = _filter_planned_specs_for_output_root(
        planned_specs=planned_specs,
        output_root=output_root_path,
    )

    return _BatchSetup(
        output_root=output_root_path,
        planned_specs=planned_specs,
        runtime_factory=runtime.runtime_factory,
        use_monitored_hf_subprocess=runtime.use_monitored_hf_subprocess,
        dry_run=dry_run,
        resume=resume,
    )


def _resolve_batch_arguments(
    *,
    output_root: str | Path,
    models: list[str],
    embedding_model: str,
    replications: int,
    config: Any | None,
) -> _BatchArguments:
    if config is None:
        return _BatchArguments(output_root, models, embedding_model, replications)
    return _BatchArguments(
        config.run.output_root,
        config.models.chat_models,
        config.models.embedding_model,
        config.run.replications,
    )


def _resolve_batch_runtime(
    *,
    runtime_factory: RuntimeFactory | None,
    dry_run: bool,
) -> _BatchRuntime:
    use_monitored_hf_subprocess = runtime_factory is None and not dry_run
    hf_runtime = _build_batch_hf_runtime(
        runtime_factory=runtime_factory,
        use_monitored_hf_subprocess=use_monitored_hf_subprocess,
    )
    profile_provider = _resolve_batch_profile_provider(
        dry_run=dry_run,
        use_monitored_hf_subprocess=use_monitored_hf_subprocess,
        hf_runtime=hf_runtime,
    )
    runtime_factory = _resolve_batch_runtime_factory(
        runtime_factory=runtime_factory,
        use_monitored_hf_subprocess=use_monitored_hf_subprocess,
        hf_runtime=hf_runtime,
    )
    return _BatchRuntime(runtime_factory, profile_provider, use_monitored_hf_subprocess)


def _build_batch_hf_runtime(
    *,
    runtime_factory: RuntimeFactory | None,
    use_monitored_hf_subprocess: bool,
) -> Any | None:
    if runtime_factory is None and not use_monitored_hf_subprocess:
        return build_runtime_factory(hf_token=_resolve_hf_token())
    return None


def _resolve_batch_profile_provider(
    *,
    dry_run: bool,
    use_monitored_hf_subprocess: bool,
    hf_runtime: Any | None,
):
    if dry_run or use_monitored_hf_subprocess:
        return default_runtime_profile_provider
    return hf_runtime.profile_for_chat_model if hf_runtime else None


def _resolve_batch_runtime_factory(
    *,
    runtime_factory: RuntimeFactory | None,
    use_monitored_hf_subprocess: bool,
    hf_runtime: Any | None,
) -> RuntimeFactory | None:
    if runtime_factory is None and not use_monitored_hf_subprocess:
        if hf_runtime is None:
            raise ValueError("Missing HF runtime for non-dry local execution.")
        return _runtime_factory_from_hf_runtime(hf_runtime)
    return runtime_factory


def _initialize_batch_state(setup: _BatchSetup) -> _BatchState:
    planned_specs = _resume_order_planned_specs_for_resume(
        planned_specs=setup.planned_specs,
        output_root=setup.output_root,
        resume=setup.resume,
        run_dir_for_spec_fn=lambda current_output_root, spec: _run_dir_for_spec(
            output_root=current_output_root,
            spec=spec,
        ),
        read_artifact_json_fn=_read_artifact_json,
    )
    total_runs = len(planned_specs)
    started_at = _status_timestamp_now()
    status_snapshot: dict[str, object]
    summary_rows: list[dict[str, object]]
    seeded_finished_run_dirs: set[Path]
    seeded_failed_run_dirs: set[Path]
    if setup.resume:
        (
            status_snapshot,
            summary_rows,
            seeded_finished_run_dirs,
            seeded_failed_run_dirs,
        ) = _build_seeded_resume_snapshot(
            output_root=setup.output_root,
            planned_specs=planned_specs,
            started_at=started_at,
        )
    else:
        summary_rows = []
        seeded_finished_run_dirs = set()
        seeded_failed_run_dirs = set()
        status_snapshot = {
            "total_runs": total_runs,
            "finished_count": 0,
            "failed_count": 0,
            "running_count": 0,
            "pending_count": total_runs,
            "failure_count": 0,
            "failures": [],
            "percent_complete": 0.0,
            "current_run": None,
            "last_completed_run": None,
            "started_at": started_at,
            "updated_at": started_at,
        }
    return _BatchState(
        total_runs=total_runs,
        planned_specs=planned_specs,
        status_snapshot=status_snapshot,
        summary_rows=summary_rows,
        persistent_sessions={},
        seeded_finished_run_dirs=seeded_finished_run_dirs,
        seeded_failed_run_dirs=seeded_failed_run_dirs,
    )


def _process_planned_spec(
    *,
    setup: _BatchSetup,
    state: _BatchState,
    spec: HFRunSpec,
) -> None:
    run_dir = _run_dir_for_spec(output_root=setup.output_root, spec=spec)
    run_dir.mkdir(parents=True, exist_ok=True)
    if _is_seeded_run(state, run_dir):
        return
    if _handle_resume_run(setup=setup, state=state, spec=spec, run_dir=run_dir):
        return
    _run_fresh_spec(setup=setup, state=state, spec=spec, run_dir=run_dir)


def _is_seeded_run(state: _BatchState, run_dir: Path) -> bool:
    return (
        run_dir in state.seeded_finished_run_dirs
        or run_dir in state.seeded_failed_run_dirs
    )


def _handle_resume_run(
    *,
    setup: _BatchSetup,
    state: _BatchState,
    spec: HFRunSpec,
    run_dir: Path,
) -> bool:
    if not setup.resume:
        return False
    _normalize_stale_running_run(run_dir)
    deferred_failure = _load_deferred_failed_summary(
        run_dir=run_dir,
        context_policy=spec.context_policy,
    )
    if deferred_failure is not None:
        _record_deferred_failure(setup=setup, state=state, failure=deferred_failure)
        return True
    cached = _load_valid_finished_summary(run_dir=run_dir, algorithm=spec.algorithm)
    if cached is None:
        return False
    _record_cached_success(setup=setup, state=state, spec=spec, summary=cached)
    return True


def _record_deferred_failure(
    *,
    setup: _BatchSetup,
    state: _BatchState,
    failure: dict[str, object],
) -> None:
    status_snapshot = state.status_snapshot
    status_snapshot["failed_count"] = _status_int(status_snapshot, "failed_count") + 1
    status_snapshot["pending_count"] = _status_int(status_snapshot, "pending_count") - 1
    failures = _status_failures(status_snapshot)
    failures.append(failure)
    status_snapshot["failures"] = failures
    status_snapshot["failure_count"] = len(failures)
    status_snapshot["updated_at"] = _status_timestamp_now()
    _write_status_snapshot(output_root=setup.output_root, status=status_snapshot)


def _record_cached_success(
    *,
    setup: _BatchSetup,
    state: _BatchState,
    spec: HFRunSpec,
    summary: dict[str, object],
) -> None:
    state.summary_rows.append(summary)
    status_snapshot = state.status_snapshot
    status_snapshot["finished_count"] = _status_int(status_snapshot, "finished_count") + 1
    status_snapshot["pending_count"] = _status_int(status_snapshot, "pending_count") - 1
    status_snapshot["last_completed_run"] = _current_run_payload_for_spec(spec)
    status_snapshot["percent_complete"] = _completion_percent(state)
    status_snapshot["updated_at"] = _status_timestamp_now()
    _write_status_snapshot(output_root=setup.output_root, status=status_snapshot)


def _run_fresh_spec(
    *,
    setup: _BatchSetup,
    state: _BatchState,
    spec: HFRunSpec,
    run_dir: Path,
) -> None:
    raw_row_path = run_dir / "raw_row.json"
    _clear_retry_artifacts(run_dir)
    _write_json(run_dir / "manifest.json", _manifest_for_spec(spec))
    _write_json(run_dir / "state.json", {"status": "running"})
    status_snapshot = state.status_snapshot
    status_snapshot["current_run"] = _current_run_payload_for_spec(spec)
    status_snapshot["running_count"] = 1
    status_snapshot["updated_at"] = _status_timestamp_now()
    _write_status_snapshot(output_root=setup.output_root, status=status_snapshot)

    try:
        runtime_result = _execute_batch_spec(setup=setup, state=state, spec=spec, run_dir=run_dir)
        if not setup.dry_run:
            _validate_structural_runtime_result(
                algorithm=spec.algorithm,
                raw_row=runtime_result["raw_row"],
            )
    except Exception as error:
        _record_batch_failure(setup=setup, state=state, spec=spec, run_dir=run_dir, error=error)
        return

    raw_row = runtime_result["raw_row"]
    _write_run_artifacts(
        run_dir=run_dir,
        spec=spec,
        runtime_result=runtime_result,
        raw_row=raw_row,
        raw_row_path=raw_row_path,
        manifest_for_spec_fn=_manifest_for_spec,
    )
    summary = _build_run_summary(
        spec=spec,
        raw_row=raw_row,
        runtime_result=runtime_result,
        raw_row_path=raw_row_path,
    )
    _write_json(run_dir / "summary.json", summary)
    state.summary_rows.append(summary)
    _record_completed_run(setup=setup, state=state, spec=spec)


def _execute_batch_spec(
    *,
    setup: _BatchSetup,
    state: _BatchState,
    spec: HFRunSpec,
    run_dir: Path,
) -> RuntimeResult:
    if setup.use_monitored_hf_subprocess:
        return _run_local_hf_spec(
            spec=spec,
            run_dir=run_dir,
            output_root=setup.output_root,
            persistent_sessions=state.persistent_sessions,
        )
    if setup.runtime_factory is None:
        raise ValueError("Missing runtime_factory for in-process execution.")
    return _execute_run(
        spec=spec,
        runtime_factory=setup.runtime_factory,
        dry_run=setup.dry_run,
        run_dir=run_dir,
    )


def _record_batch_failure(
    *,
    setup: _BatchSetup,
    state: _BatchState,
    spec: HFRunSpec,
    run_dir: Path,
    error: Exception,
) -> None:
    failure_payload = {
        "type": type(error).__name__,
        "message": str(error),
        "status": "failed",
    }
    failure_kind = _classify_failure_payload(failure_payload)
    should_retry_on_resume = _should_keep_failure_pending_on_resume(
        resume=setup.resume,
        failure_kind=failure_kind,
        context_policy=spec.context_policy,
    )
    _write_json(run_dir / "error.json", failure_payload)
    _write_json(run_dir / "state.json", {"status": "failed"})
    status_snapshot = state.status_snapshot
    status_snapshot["running_count"] = 0
    status_snapshot["current_run"] = None
    if should_retry_on_resume:
        status_snapshot["updated_at"] = _status_timestamp_now()
        _write_status_snapshot(output_root=setup.output_root, status=status_snapshot)
    else:
        failures = _status_failures(status_snapshot)
        failure_entry: dict[str, object] = {
            "run_dir": str(run_dir),
            "message": str(error),
            "type": type(error).__name__,
        }
        status_snapshot["failed_count"] = _status_int(status_snapshot, "failed_count") + 1
        status_snapshot["pending_count"] = _status_int(status_snapshot, "pending_count") - 1
        failures.append(failure_entry)
        status_snapshot["failures"] = failures
        status_snapshot["failure_count"] = len(failures)
        status_snapshot["updated_at"] = _status_timestamp_now()
        _write_status_snapshot(output_root=setup.output_root, status=status_snapshot)
    if failure_kind == "infrastructure":
        raise BatchInfrastructureFailure(
            f"Infrastructure failure while executing {run_dir}: {error}"
        ) from error


def _record_completed_run(*, setup: _BatchSetup, state: _BatchState, spec: HFRunSpec) -> None:
    status_snapshot = state.status_snapshot
    status_snapshot["running_count"] = 0
    status_snapshot["current_run"] = None
    status_snapshot["last_completed_run"] = _current_run_payload_for_spec(spec)
    status_snapshot["finished_count"] = _status_int(status_snapshot, "finished_count") + 1
    status_snapshot["pending_count"] = _status_int(status_snapshot, "pending_count") - 1
    status_snapshot["percent_complete"] = _completion_percent(state)
    status_snapshot["updated_at"] = _status_timestamp_now()
    _write_status_snapshot(output_root=setup.output_root, status=status_snapshot)


def _current_run_payload_for_spec(spec: HFRunSpec) -> dict[str, object]:
    return _current_run_payload(
        algorithm=spec.algorithm,
        model=spec.model,
        decoding_algorithm=spec.decoding.algorithm,
        graph_source=spec.graph_source,
        pair_name=spec.pair_name,
        condition_bits=spec.condition_bits,
        replication=spec.replication,
    )


def _completion_percent(state: _BatchState) -> float:
    finished_count = _status_int(state.status_snapshot, "finished_count")
    return round((finished_count / state.total_runs) * 100.0, 2)


def _finalize_batch_outputs(*, setup: _BatchSetup, state: _BatchState) -> None:
    summary_frame = pd.DataFrame.from_records(state.summary_rows)
    summary_frame.to_csv(setup.output_root / "batch_summary.csv", index=False)
    if setup.dry_run or summary_frame.empty:
        return
    write_aggregated_outputs(setup.output_root, summary_frame)


def _build_seeded_resume_snapshot(
    *,
    output_root: Path,
    planned_specs: list[HFRunSpec],
    started_at: str,
) -> tuple[dict[str, object], list[dict[str, object]], set[Path], set[Path]]:
    return _resume_build_seeded_resume_snapshot(
        output_root=output_root,
        planned_specs=planned_specs,
        started_at=started_at,
        run_dir_for_spec_fn=lambda current_output_root, spec: _run_dir_for_spec(
            output_root=current_output_root,
            spec=spec,
        ),
        current_run_payload_fn=_current_run_payload,
        status_timestamp_now_fn=_status_timestamp_now,
        validate_structural_runtime_result_fn=_validate_structural_runtime_result,
        normalize_stale_running_run_fn=_normalize_stale_running_run,
        read_artifact_json_fn=_read_artifact_json,
        write_json_fn=_write_json,
    )


def run_single_spec(
    *,
    spec: HFRunSpec,
    output_root: str | Path,
    runtime_factory: RuntimeFactory | None = None,
    dry_run: bool = False,
    resume: bool = False,
) -> dict[str, object]:
    setup = _prepare_single_spec_setup(
        spec=spec,
        output_root=output_root,
        runtime_factory=runtime_factory,
        dry_run=dry_run,
    )
    cached_summary = _load_cached_single_spec_summary(setup=setup, resume=resume)
    if cached_summary is not None:
        return cached_summary

    _prepare_single_spec_run_directory(setup)
    try:
        runtime_result = _execute_single_spec(setup)
    except Exception as error:
        _record_single_spec_failure(setup=setup, error=error)
        raise
    finally:
        _close_persistent_sessions(setup.persistent_sessions)
    return _complete_single_spec(setup=setup, runtime_result=runtime_result)


@dataclass
class _SingleSpecSetup:
    spec: HFRunSpec
    output_root: Path
    run_dir: Path
    summary_path: Path
    raw_row_path: Path
    runtime_factory: RuntimeFactory | None
    persistent_sessions: dict[str, PersistentHFWorkerSession]
    use_monitored_hf_subprocess: bool
    dry_run: bool


def _prepare_single_spec_setup(
    *,
    spec: HFRunSpec,
    output_root: str | Path,
    runtime_factory: RuntimeFactory | None,
    dry_run: bool,
) -> _SingleSpecSetup:
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    run_dir = _run_dir_for_spec(output_root=output_root_path, spec=spec)
    run_dir.mkdir(parents=True, exist_ok=True)
    return _SingleSpecSetup(
        spec=spec,
        output_root=output_root_path,
        run_dir=run_dir,
        summary_path=run_dir / "summary.json",
        raw_row_path=run_dir / "raw_row.json",
        runtime_factory=runtime_factory,
        persistent_sessions={},
        use_monitored_hf_subprocess=runtime_factory is None and not dry_run,
        dry_run=dry_run,
    )


def _load_cached_single_spec_summary(
    *,
    setup: _SingleSpecSetup,
    resume: bool,
) -> dict[str, object] | None:
    if resume:
        _normalize_stale_running_run(setup.run_dir)
    cached_summary = _load_valid_finished_summary(
        run_dir=setup.run_dir,
        algorithm=setup.spec.algorithm,
    )
    if not resume or cached_summary is None:
        return None
    _write_smoke_verdict(
        output_root=setup.output_root,
        run_dir=setup.run_dir,
        spec_identity=_smoke_spec_identity(setup.spec),
        status="success",
        worker_loaded_model=True,
    )
    return cached_summary


def _prepare_single_spec_run_directory(setup: _SingleSpecSetup) -> None:
    _clear_retry_artifacts(setup.run_dir)
    _write_json(setup.run_dir / "manifest.json", _manifest_for_spec(setup.spec))
    _write_json(setup.run_dir / "state.json", {"status": "running"})


def _execute_single_spec(setup: _SingleSpecSetup) -> RuntimeResult:
    if setup.use_monitored_hf_subprocess:
        runtime_result = _run_local_hf_spec(
            spec=setup.spec,
            run_dir=setup.run_dir,
            output_root=setup.output_root,
            persistent_sessions=setup.persistent_sessions,
        )
    else:
        runtime_factory = setup.runtime_factory
        if runtime_factory is None:
            hf_runtime = build_runtime_factory(hf_token=_resolve_hf_token())
            runtime_factory = _runtime_factory_from_hf_runtime(hf_runtime)
        runtime_result = _execute_run(
            spec=setup.spec,
            runtime_factory=runtime_factory,
            dry_run=setup.dry_run,
            run_dir=setup.run_dir,
        )
    if not setup.dry_run:
        _validate_structural_runtime_result(
            algorithm=setup.spec.algorithm,
            raw_row=runtime_result["raw_row"],
        )
    return runtime_result


def _record_single_spec_failure(*, setup: _SingleSpecSetup, error: Exception) -> None:
    _write_json(
        setup.run_dir / "error.json",
        {
            "type": type(error).__name__,
            "message": str(error),
            "status": "failed",
        },
    )
    _write_json(setup.run_dir / "state.json", {"status": "failed"})
    _write_smoke_verdict(
        output_root=setup.output_root,
        run_dir=setup.run_dir,
        spec_identity=_smoke_spec_identity(setup.spec),
        status="failed",
        failure_type=type(error).__name__,
        failure_message=str(error),
        worker_loaded_model=_worker_loaded_model(setup.run_dir),
    )


def _close_persistent_sessions(
    persistent_sessions: dict[str, PersistentHFWorkerSession],
) -> None:
    for session in persistent_sessions.values():
        session.close()


def _complete_single_spec(
    *,
    setup: _SingleSpecSetup,
    runtime_result: RuntimeResult,
) -> dict[str, object]:
    raw_row = runtime_result["raw_row"]
    _write_run_artifacts(
        run_dir=setup.run_dir,
        spec=setup.spec,
        runtime_result=runtime_result,
        raw_row=raw_row,
        raw_row_path=setup.raw_row_path,
        manifest_for_spec_fn=_manifest_for_spec,
    )

    summary = _build_run_summary(
        spec=setup.spec,
        raw_row=raw_row,
        runtime_result=runtime_result,
        raw_row_path=setup.raw_row_path,
    )
    _write_json(setup.summary_path, summary)
    _write_smoke_verdict(
        output_root=setup.output_root,
        run_dir=setup.run_dir,
        spec_identity=_smoke_spec_identity(setup.spec),
        status="success",
        worker_loaded_model=_worker_loaded_model(setup.run_dir) or True,
    )
    return summary
