from __future__ import annotations

from llm_conceptual_modeling.hf_experiments import (
    _build_worker_command,
    _coerce_timeout_seconds,
    _collect_resume_history,
    _is_finished_run_directory,
    _is_retryable_worker_error,
    _load_deferred_failed_summary,
    _resolve_resume_pass_mode,
    _resolve_run_retry_attempts,
    _resolve_stage_timeout_seconds,
    _resolve_startup_timeout_seconds,
    _resolve_worker_process_mode,
    _run_algo1,
    _run_algo2,
    _run_algo3,
    _run_local_hf_spec,
    _run_local_hf_spec_subprocess,
    _status_failures,
    _status_int,
)


def test_hf_experiments_helpers_are_package_aliases() -> None:
    assert _is_finished_run_directory.__module__ == "llm_conceptual_modeling.hf_state.resume_state"
    assert (
        _load_deferred_failed_summary.__module__
        == "llm_conceptual_modeling.hf_state.resume_state"
    )
    assert _resolve_resume_pass_mode.__module__ == "llm_conceptual_modeling.hf_state.resume_state"
    assert _resolve_run_retry_attempts.__module__ == "llm_conceptual_modeling.hf_execution.helpers"
    assert _resolve_startup_timeout_seconds.__module__ == "llm_conceptual_modeling.hf_execution.helpers"
    assert _resolve_stage_timeout_seconds.__module__ == "llm_conceptual_modeling.hf_execution.helpers"
    assert _resolve_worker_process_mode.__module__ == "llm_conceptual_modeling.hf_execution.helpers"
    assert _build_worker_command.__module__ == "llm_conceptual_modeling.hf_execution.helpers"
    assert (
        _run_local_hf_spec_subprocess.__module__
        == "llm_conceptual_modeling.hf_execution.runtime"
    )
    assert _run_local_hf_spec.__module__ == "llm_conceptual_modeling.hf_execution.runtime"
    assert _is_retryable_worker_error.__module__ == "llm_conceptual_modeling.hf_execution.helpers"
    assert _coerce_timeout_seconds.__module__ == "llm_conceptual_modeling.hf_execution.helpers"
    assert _collect_resume_history.__module__ == "llm_conceptual_modeling.hf_state.resume_state"
    assert _status_int.__module__ == "llm_conceptual_modeling.hf_state.resume_state"
    assert _status_failures.__module__ == "llm_conceptual_modeling.hf_state.resume_state"
    assert _run_algo1.__module__ == "llm_conceptual_modeling.hf_pipeline.algo1"
    assert _run_algo2.__module__ == "llm_conceptual_modeling.hf_pipeline.algo2"
    assert _run_algo3.__module__ == "llm_conceptual_modeling.hf_pipeline.algo3"
