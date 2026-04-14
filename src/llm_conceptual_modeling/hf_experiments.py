from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, cast

import pandas as pd

from llm_conceptual_modeling.common.hf_transformers import (
    DecodingConfig,
    HFTransformersRuntimeFactory,
    RuntimeProfile,
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
    plan_paper_batch_specs,
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
from llm_conceptual_modeling.hf_batch.spec_path import (
    spec_identity as _spec_identity,
)
from llm_conceptual_modeling.hf_batch.prompts import (
    build_prompt_bundle as _build_prompt_bundle,  # noqa: F401
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
    write_smoke_verdict as _write_smoke_verdict,
)
from llm_conceptual_modeling.hf_batch.types import HFRunSpec, RuntimeFactory, RuntimeResult
from llm_conceptual_modeling.hf_batch.utils import (
    manifest_for_spec as _manifest_for_spec,
)
from llm_conceptual_modeling.hf_batch.utils import (
    resolve_hf_token as _resolve_hf_token,
)
from llm_conceptual_modeling.hf_batch.utils import (
    slugify_model as _slugify_model,
)
from llm_conceptual_modeling.hf_batch.utils import (
    write_json as _write_json,
)
from llm_conceptual_modeling.hf_batch.utils import (
    write_text as _write_text,
)
from llm_conceptual_modeling.hf_execution.helpers import (
    build_worker_command as _execution_build_worker_command,
)
from llm_conceptual_modeling.hf_execution.helpers import (
    coerce_timeout_seconds as _execution_coerce_timeout_seconds,
)
from llm_conceptual_modeling.hf_execution.helpers import (
    is_retryable_worker_error as _execution_is_retryable_worker_error,
)
from llm_conceptual_modeling.hf_execution.helpers import (
    resolve_max_requests_per_worker_process as _execution_resolve_max_requests_per_worker_process,
)
from llm_conceptual_modeling.hf_execution.helpers import (
    resolve_run_retry_attempts as _execution_resolve_run_retry_attempts,
)
from llm_conceptual_modeling.hf_execution.helpers import (
    resolve_stage_timeout_seconds as _execution_resolve_stage_timeout_seconds,
)
from llm_conceptual_modeling.hf_execution.helpers import (
    resolve_startup_timeout_seconds as _execution_resolve_startup_timeout_seconds,
)
from llm_conceptual_modeling.hf_execution.helpers import (
    resolve_worker_process_mode as _execution_resolve_worker_process_mode,
)
from llm_conceptual_modeling.hf_execution.runtime import (
    run_local_hf_spec as _execution_run_local_hf_spec,
)
from llm_conceptual_modeling.hf_execution.runtime import (
    run_local_hf_spec_subprocess as _execution_run_local_hf_spec_subprocess,
)
from llm_conceptual_modeling.hf_execution.subprocess import (
    run_monitored_command as _run_monitored_command,
)
from llm_conceptual_modeling.hf_pipeline.algo1 import run_algo1 as _pipeline_run_algo1
from llm_conceptual_modeling.hf_pipeline.algo2 import run_algo2 as _pipeline_run_algo2
from llm_conceptual_modeling.hf_pipeline.algo3 import run_algo3 as _pipeline_run_algo3
from llm_conceptual_modeling.hf_pipeline.metrics import (
    connection_metric_summary as _pipeline_connection_metric_summary,
)
from llm_conceptual_modeling.hf_pipeline.metrics import (
    sanitize_algorithm_edge_result as _pipeline_sanitize_algorithm_edge_result,
)
from llm_conceptual_modeling.hf_pipeline.metrics import (
    summary_from_raw_row as _pipeline_summary_from_raw_row,
)
from llm_conceptual_modeling.hf_pipeline.metrics import (
    trace_metric_summary as _pipeline_trace_metric_summary,
)
from llm_conceptual_modeling.hf_pipeline.metrics import (
    validate_structural_runtime_result as _pipeline_validate_structural_runtime_result,
)
from llm_conceptual_modeling.hf_run_config import HFRunConfig
from llm_conceptual_modeling.hf_state.resume_state import (
    build_seeded_resume_snapshot as _resume_build_seeded_resume_snapshot,
)
from llm_conceptual_modeling.hf_state.resume_state import (
    classify_failure_payload as _resume_classify_failure_payload,
)
from llm_conceptual_modeling.hf_state.resume_state import (
    collect_resume_history as _resume_collect_resume_history,
)
from llm_conceptual_modeling.hf_state.resume_state import (
    is_finished_run_directory as _resume_is_finished_run_directory,
)
from llm_conceptual_modeling.hf_state.resume_state import (
    load_deferred_failed_summary as _resume_load_deferred_failed_summary,
)
from llm_conceptual_modeling.hf_state.resume_state import (
    load_valid_finished_summary as _resume_load_valid_finished_summary,
)
from llm_conceptual_modeling.hf_state.resume_state import (
    order_planned_specs_for_resume as _resume_order_planned_specs_for_resume,
)
from llm_conceptual_modeling.hf_state.resume_state import (
    resolve_resume_pass_mode as _resume_resolve_resume_pass_mode,
)
from llm_conceptual_modeling.hf_state.resume_state import (
    resolve_retry_infrastructure_failures_on_resume,
    resolve_retry_oom_failures_on_resume,
    resolve_retry_structural_failures_on_resume,
)
from llm_conceptual_modeling.hf_state.resume_state import (
    resolve_retry_timeout_failures_on_resume as _resume_resolve_retry_timeout_failures_on_resume,
)
from llm_conceptual_modeling.hf_state.resume_state import (
    status_failures as _resume_status_failures,
)
from llm_conceptual_modeling.hf_state.resume_state import (
    status_int as _resume_status_int,
)
from llm_conceptual_modeling.hf_worker.persistent import PersistentHFWorkerSession
from llm_conceptual_modeling.hf_worker.state import (
    mark_worker_ready_for_execution as _worker_state_mark_ready_for_execution,
)

_resume_resolve_retry_infrastructure_failures_on_resume = (
    resolve_retry_infrastructure_failures_on_resume
)
_resume_resolve_retry_oom_failures_on_resume = resolve_retry_oom_failures_on_resume
_resume_resolve_retry_structural_failures_on_resume = (
    resolve_retry_structural_failures_on_resume
)

_connection_metric_summary = _pipeline_connection_metric_summary
_trace_metric_summary = _pipeline_trace_metric_summary
_summary_from_raw_row = _pipeline_summary_from_raw_row
_validate_structural_runtime_result = _pipeline_validate_structural_runtime_result
_sanitize_algorithm_edge_result = _pipeline_sanitize_algorithm_edge_result
_is_finished_run_directory = _resume_is_finished_run_directory
_load_valid_finished_summary = _resume_load_valid_finished_summary
_load_deferred_failed_summary = _resume_load_deferred_failed_summary
_status_int = _resume_status_int
_status_failures = _resume_status_failures
_resolve_retry_timeout_failures_on_resume = _resume_resolve_retry_timeout_failures_on_resume
_resolve_retry_oom_failures_on_resume = _resume_resolve_retry_oom_failures_on_resume
_resolve_retry_infrastructure_failures_on_resume = (
    _resume_resolve_retry_infrastructure_failures_on_resume
)
_resolve_retry_structural_failures_on_resume = _resume_resolve_retry_structural_failures_on_resume
_resolve_resume_pass_mode = _resume_resolve_resume_pass_mode
_classify_failure_payload = _resume_classify_failure_payload
_build_worker_command = _execution_build_worker_command
_run_local_hf_spec_subprocess = _execution_run_local_hf_spec_subprocess
_run_local_hf_spec = _execution_run_local_hf_spec
_resolve_startup_timeout_seconds = _execution_resolve_startup_timeout_seconds
_resolve_stage_timeout_seconds = _execution_resolve_stage_timeout_seconds
_resolve_run_retry_attempts = _execution_resolve_run_retry_attempts
_resolve_worker_process_mode = _execution_resolve_worker_process_mode
_resolve_max_requests_per_worker_process = _execution_resolve_max_requests_per_worker_process
_is_retryable_worker_error = _execution_is_retryable_worker_error
_coerce_timeout_seconds = _execution_coerce_timeout_seconds
_collect_resume_history = _resume_collect_resume_history
_run_monitored_command = _run_monitored_command
_run_algo1 = _pipeline_run_algo1
_run_algo2 = _pipeline_run_algo2
_run_algo3 = _pipeline_run_algo3
run_monitored_command = _run_monitored_command


class BatchInfrastructureFailure(RuntimeError):
    """Abort a batch when the worker host is unhealthy."""


def plan_paper_batch(
    *,
    models: list[str],
    embedding_model: str,
    replications: int,
    algorithms: tuple[str, ...] | None = None,
    config: HFRunConfig | None = None,
    runtime_profile_provider: Callable[[str], RuntimeProfile] | None = None,
) -> list[HFRunSpec]:
    return plan_paper_batch_specs(
        models=models,
        embedding_model=embedding_model,
        replications=replications,
        algorithms=algorithms,
        config=config,
        runtime_profile_provider=runtime_profile_provider,
    )


def run_paper_batch(
    *,
    output_root: str | Path,
    models: list[str],
    embedding_model: str,
    replications: int,
    algorithms: tuple[str, ...] | None = None,
    config: HFRunConfig | None = None,
    runtime_factory: RuntimeFactory | None = None,
    resume: bool = False,
    dry_run: bool = False,
) -> None:
    if config is not None:
        output_root = config.run.output_root
        models = config.models.chat_models
        embedding_model = config.models.embedding_model
        replications = config.run.replications
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    use_monitored_hf_subprocess = runtime_factory is None and not dry_run
    hf_runtime = None
    if runtime_factory is None and not use_monitored_hf_subprocess:
        hf_runtime = build_runtime_factory(hf_token=_resolve_hf_token())
    if dry_run or use_monitored_hf_subprocess:
        profile_provider = default_runtime_profile_provider
    else:
        profile_provider = hf_runtime.profile_for_chat_model if hf_runtime else None
    planned_specs = plan_paper_batch(
        models=models,
        embedding_model=embedding_model,
        replications=replications,
        algorithms=algorithms,
        config=config,
        runtime_profile_provider=profile_provider,
    )
    planned_specs = _filter_planned_specs_for_output_root(
        planned_specs=planned_specs,
        output_root=output_root_path,
    )
    if runtime_factory is None and not use_monitored_hf_subprocess:
        if hf_runtime is None:
            raise ValueError("Missing HF runtime for non-dry local execution.")
        runtime_factory = _runtime_factory_from_hf_runtime(hf_runtime)

    summary_rows: list[dict[str, object]] = []
    persistent_sessions: dict[str, PersistentHFWorkerSession] = {}
    seeded_finished_run_dirs: set[Path] = set()
    seeded_failed_run_dirs: set[Path] = set()
    planned_specs = _resume_order_planned_specs_for_resume(
        planned_specs=planned_specs,
        output_root=output_root_path,
        resume=resume,
        run_dir_for_spec_fn=lambda current_output_root, spec: _run_dir_for_spec(
            output_root=current_output_root,
            spec=spec,
        ),
        read_artifact_json_fn=_read_artifact_json,
    )
    total_runs = len(planned_specs)
    started_at = _status_timestamp_now()
    last_completed_run: dict[str, object] | None = None
    status_snapshot: dict[str, object]
    if resume:
        (
            status_snapshot,
            summary_rows,
            seeded_finished_run_dirs,
            seeded_failed_run_dirs,
        ) = _build_seeded_resume_snapshot(
            output_root=output_root_path,
            planned_specs=planned_specs,
            started_at=started_at,
        )
        last_completed_run = cast(dict[str, object] | None, status_snapshot["last_completed_run"])
    else:
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
    _write_status_snapshot(output_root=output_root_path, status=status_snapshot)

    try:
        for spec in planned_specs:
            run_dir = _run_dir_for_spec(output_root=output_root_path, spec=spec)
            run_dir.mkdir(parents=True, exist_ok=True)
            summary_path = run_dir / "summary.json"
            raw_row_path = run_dir / "raw_row.json"

            if resume and run_dir in seeded_finished_run_dirs:
                continue
            if resume and run_dir in seeded_failed_run_dirs:
                continue

            if resume:
                _normalize_stale_running_run(run_dir)
                deferred_failure = _load_deferred_failed_summary(
                    run_dir=run_dir,
                    context_policy=spec.context_policy,
                )
                if deferred_failure is not None:
                    failed_count = _status_int(status_snapshot, "failed_count")
                    pending_count = _status_int(status_snapshot, "pending_count")
                    status_snapshot["failed_count"] = failed_count + 1
                    status_snapshot["pending_count"] = pending_count - 1
                    failures = _status_failures(status_snapshot)
                    failures.append(deferred_failure)
                    status_snapshot["failures"] = failures
                    status_snapshot["failure_count"] = len(failures)
                    status_snapshot["updated_at"] = _status_timestamp_now()
                    _write_status_snapshot(output_root=output_root_path, status=status_snapshot)
                    continue
            cached = _load_valid_finished_summary(run_dir=run_dir, algorithm=spec.algorithm)
            if resume and cached is not None:
                summary_rows.append(cached)
                last_completed_run = _current_run_payload(
                    algorithm=spec.algorithm,
                    model=spec.model,
                    decoding_algorithm=spec.decoding.algorithm,
                    pair_name=spec.pair_name,
                    condition_bits=spec.condition_bits,
                    replication=spec.replication,
                )
                finished_count = _status_int(status_snapshot, "finished_count")
                pending_count = _status_int(status_snapshot, "pending_count")
                status_snapshot["finished_count"] = finished_count + 1
                status_snapshot["pending_count"] = pending_count - 1
                status_snapshot["last_completed_run"] = last_completed_run
                status_snapshot["percent_complete"] = round(
                    (_status_int(status_snapshot, "finished_count") / total_runs) * 100.0,
                    2,
                )
                status_snapshot["updated_at"] = _status_timestamp_now()
                _write_status_snapshot(output_root=output_root_path, status=status_snapshot)
                continue

            _clear_retry_artifacts(run_dir)
            _write_json(run_dir / "manifest.json", _manifest_for_spec(spec))
            _write_json(run_dir / "state.json", {"status": "running"})
            current_run = _current_run_payload(
                algorithm=spec.algorithm,
                model=spec.model,
                decoding_algorithm=spec.decoding.algorithm,
                pair_name=spec.pair_name,
                condition_bits=spec.condition_bits,
                replication=spec.replication,
            )
            status_snapshot["current_run"] = current_run
            status_snapshot["running_count"] = 1
            status_snapshot["updated_at"] = _status_timestamp_now()
            _write_status_snapshot(output_root=output_root_path, status=status_snapshot)

            try:
                if use_monitored_hf_subprocess:
                    runtime_result = _run_local_hf_spec(
                        spec=spec,
                        run_dir=run_dir,
                        output_root=output_root_path,
                        persistent_sessions=persistent_sessions,
                    )
                else:
                    if runtime_factory is None:
                        raise ValueError("Missing runtime_factory for in-process execution.")
                    runtime_result = _execute_run(
                        spec=spec,
                        runtime_factory=runtime_factory,
                        dry_run=dry_run,
                        run_dir=run_dir,
                    )
                if not dry_run:
                    _validate_structural_runtime_result(
                        algorithm=spec.algorithm,
                        raw_row=runtime_result["raw_row"],
                    )
            except Exception as error:
                failure_payload = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "status": "failed",
                }
                failure_kind = _classify_failure_payload(failure_payload)
                should_retry_on_resume = _should_keep_failure_pending_on_resume(
                    resume=resume,
                    failure_kind=failure_kind,
                    context_policy=spec.context_policy,
                )
                _write_json(run_dir / "error.json", failure_payload)
                _write_json(run_dir / "state.json", {"status": "failed"})
                status_snapshot["running_count"] = 0
                status_snapshot["current_run"] = None
                if should_retry_on_resume:
                    status_snapshot["updated_at"] = _status_timestamp_now()
                    _write_status_snapshot(output_root=output_root_path, status=status_snapshot)
                else:
                    failures = _status_failures(status_snapshot)
                    failure_entry = {
                        "run_dir": str(run_dir),
                        "message": str(error),
                        "type": type(error).__name__,
                    }
                    status_snapshot["failed_count"] = _status_int(
                        status_snapshot,
                        "failed_count",
                    ) + 1
                    status_snapshot["pending_count"] = _status_int(
                        status_snapshot,
                        "pending_count",
                    ) - 1
                    failures.append(failure_entry)
                    status_snapshot["failures"] = failures
                    status_snapshot["failure_count"] = len(failures)
                    status_snapshot["updated_at"] = _status_timestamp_now()
                    _write_status_snapshot(output_root=output_root_path, status=status_snapshot)
                if failure_kind == "infrastructure":
                    raise BatchInfrastructureFailure(
                        f"Infrastructure failure while executing {run_dir}: {error}"
                    ) from error
                continue

            raw_row = runtime_result["raw_row"]
            _write_json(raw_row_path, raw_row)
            _write_json(run_dir / "state.json", {"status": "finished"})
            _write_json(run_dir / "runtime.json", runtime_result["runtime"])
            _write_text(run_dir / "raw_response.json", runtime_result["raw_response"])

            summary = {
                "algorithm": spec.algorithm,
                "model": spec.model,
                "embedding_model": spec.embedding_model,
                "decoding_algorithm": spec.decoding.algorithm,
                "condition_label": spec.condition_label,
                "pair_name": spec.pair_name,
                "condition_bits": spec.condition_bits,
                "replication": spec.replication,
                "status": "finished",
                "thinking_mode_supported": runtime_result["runtime"].get(
                    "thinking_mode_supported", False
                ),
                "raw_row_path": str(raw_row_path),
                **_summary_from_raw_row(spec.algorithm, raw_row),
                **runtime_result.get("summary", {}),
            }
            _write_json(summary_path, summary)
            summary_rows.append(summary)
            last_completed_run = current_run
            status_snapshot["running_count"] = 0
            status_snapshot["current_run"] = None
            status_snapshot["last_completed_run"] = last_completed_run
            status_snapshot["finished_count"] = _status_int(status_snapshot, "finished_count") + 1
            status_snapshot["pending_count"] = _status_int(status_snapshot, "pending_count") - 1
            status_snapshot["percent_complete"] = round(
                (_status_int(status_snapshot, "finished_count") / total_runs) * 100.0,
                2,
            )
            status_snapshot["updated_at"] = _status_timestamp_now()
            _write_status_snapshot(output_root=output_root_path, status=status_snapshot)
    finally:
        for session in persistent_sessions.values():
            session.close()

    summary_frame = pd.DataFrame.from_records(summary_rows)
    summary_frame.to_csv(output_root_path / "batch_summary.csv", index=False)
    if dry_run:
        return
    if summary_frame.empty:
        return

    write_aggregated_outputs(output_root_path, summary_frame)


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


def select_run_spec(
    *,
    config: HFRunConfig,
    algorithm: str,
    model: str,
    pair_name: str,
    condition_bits: str,
    decoding: DecodingConfig,
    replication: int,
    runtime_profile_provider: Callable[[str], RuntimeProfile] | None = None,
) -> HFRunSpec:
    specs = plan_paper_batch(
        models=[model],
        embedding_model=config.models.embedding_model,
        replications=max(replication + 1, 1),
        algorithms=(algorithm,),
        config=config,
        runtime_profile_provider=runtime_profile_provider,
    )
    for spec in specs:
        if (
            spec.algorithm == algorithm
            and spec.model == model
            and spec.pair_name == pair_name
            and spec.condition_bits == condition_bits
            and spec.replication == replication
            and spec.decoding == decoding
        ):
            return spec
    raise ValueError(
        "No configured run spec matches "
        f"algorithm={algorithm!r}, model={model!r}, pair_name={pair_name!r}, "
        f"condition_bits={condition_bits!r}, decoding={decoding!r}, "
        f"replication={replication!r}."
    )


def run_single_spec(
    *,
    spec: HFRunSpec,
    output_root: str | Path,
    runtime_factory: RuntimeFactory | None = None,
    dry_run: bool = False,
    resume: bool = False,
) -> dict[str, object]:
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    persistent_sessions: dict[str, PersistentHFWorkerSession] = {}

    use_monitored_hf_subprocess = runtime_factory is None and not dry_run

    run_dir = _run_dir_for_spec(output_root=output_root_path, spec=spec)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    raw_row_path = run_dir / "raw_row.json"

    if resume:
        _normalize_stale_running_run(run_dir)
    cached_summary = _load_valid_finished_summary(run_dir=run_dir, algorithm=spec.algorithm)
    if resume and cached_summary is not None:
        _write_smoke_verdict(
            output_root=output_root_path,
            run_dir=run_dir,
            spec_identity=_smoke_spec_identity(spec),
            status="success",
            worker_loaded_model=True,
        )
        return cached_summary

    _clear_retry_artifacts(run_dir)
    _write_json(run_dir / "manifest.json", _manifest_for_spec(spec))
    _write_json(run_dir / "state.json", {"status": "running"})
    try:
        if use_monitored_hf_subprocess:
            runtime_result = _run_local_hf_spec(
                spec=spec,
                run_dir=run_dir,
                output_root=output_root_path,
                persistent_sessions=persistent_sessions,
            )
        else:
            if runtime_factory is None:
                hf_runtime = build_runtime_factory(hf_token=_resolve_hf_token())
                runtime_factory = _runtime_factory_from_hf_runtime(hf_runtime)
            runtime_result = _execute_run(
                spec=spec,
                runtime_factory=runtime_factory,
                dry_run=dry_run,
                run_dir=run_dir,
            )
        if not dry_run:
            _validate_structural_runtime_result(
                algorithm=spec.algorithm,
                raw_row=runtime_result["raw_row"],
            )
    except Exception as error:
        _write_json(
            run_dir / "error.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "status": "failed",
            },
        )
        _write_json(run_dir / "state.json", {"status": "failed"})
        _write_smoke_verdict(
            output_root=output_root_path,
            run_dir=run_dir,
            spec_identity=_smoke_spec_identity(spec),
            status="failed",
            failure_type=type(error).__name__,
            failure_message=str(error),
            worker_loaded_model=_worker_loaded_model(run_dir),
        )
        raise
    finally:
        for session in persistent_sessions.values():
            session.close()
    raw_row = runtime_result["raw_row"]
    _write_json(raw_row_path, raw_row)
    _write_json(run_dir / "state.json", {"status": "finished"})
    _write_json(run_dir / "runtime.json", runtime_result["runtime"])
    _write_text(run_dir / "raw_response.json", runtime_result["raw_response"])

    summary = {
        "algorithm": spec.algorithm,
        "model": spec.model,
        "embedding_model": spec.embedding_model,
        "decoding_algorithm": spec.decoding.algorithm,
        "condition_label": spec.condition_label,
        "pair_name": spec.pair_name,
        "condition_bits": spec.condition_bits,
        "replication": spec.replication,
        "status": "finished",
        "thinking_mode_supported": runtime_result["runtime"].get(
            "thinking_mode_supported", False
        ),
        "raw_row_path": str(raw_row_path),
        **_summary_from_raw_row(spec.algorithm, raw_row),
        **runtime_result.get("summary", {}),
    }
    _write_json(summary_path, summary)
    _write_smoke_verdict(
        output_root=output_root_path,
        run_dir=run_dir,
        spec_identity=_smoke_spec_identity(spec),
        status="success",
        worker_loaded_model=_worker_loaded_model(run_dir) or True,
    )
    return summary


def _worker_loaded_model(run_dir: Path) -> bool:
    worker_state = _read_artifact_json(run_dir / "worker_state.json")
    return worker_state.get("model_loaded") is True or (run_dir / "active_stage.json").exists()


def _mark_worker_ready_for_execution(run_dir: Path | None) -> None:
    if run_dir is None:
        return
    _worker_state_mark_ready_for_execution(
        run_dir / "worker_state.json",
        timestamp=_status_timestamp_now(),
    )


def _execute_run(
    *,
    spec: HFRunSpec,
    runtime_factory: RuntimeFactory,
    dry_run: bool,
    run_dir: Path | None = None,
) -> RuntimeResult:
    if dry_run:
        return {
            "raw_row": dict(spec.raw_context),
            "runtime": {
                "thinking_mode_supported": spec.runtime_profile.supports_thinking_toggle,
                "device": spec.runtime_profile.device,
                "dtype": spec.runtime_profile.dtype,
                "quantization": spec.runtime_profile.quantization,
            },
            "raw_response": "[]",
        }
    try:
        runtime_callable = cast(Any, runtime_factory)
        return runtime_callable(spec, run_dir=run_dir)
    except TypeError as error:
        if "run_dir" not in str(error):
            raise
        return runtime_factory(spec)




def _should_keep_failure_pending_on_resume(
    *,
    resume: bool,
    failure_kind: str,
    context_policy: dict[str, object] | None,
) -> bool:
    if not resume:
        return False
    retry_checks = {
        "timeout": _resolve_retry_timeout_failures_on_resume,
        "oom": _resolve_retry_oom_failures_on_resume,
        "infrastructure": _resolve_retry_infrastructure_failures_on_resume,
        "structural": _resolve_retry_structural_failures_on_resume,
    }
    retry_check = retry_checks.get(failure_kind)
    if retry_check is None:
        return False
    return retry_check(context_policy)


def _runtime_factory_from_hf_runtime(hf_runtime: HFTransformersRuntimeFactory) -> RuntimeFactory:
    def runtime(spec: HFRunSpec, *, run_dir: Path | None = None) -> RuntimeResult:
        if spec.algorithm == "algo1":
            return _run_algo1(spec, hf_runtime=hf_runtime, run_dir=run_dir)
        if spec.algorithm == "algo2":
            return _run_algo2(spec, hf_runtime=hf_runtime, run_dir=run_dir)
        if spec.algorithm == "algo3":
            return _run_algo3(spec, hf_runtime=hf_runtime, run_dir=run_dir)
        raise ValueError(f"Unsupported algorithm: {spec.algorithm}")

    return runtime

