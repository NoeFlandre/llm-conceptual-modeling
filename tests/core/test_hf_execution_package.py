from llm_conceptual_modeling.hf_execution import (
    build_worker_command,
    execute_run,
    is_retryable_worker_error,
    resolve_max_requests_per_worker_process,
    resolve_worker_process_mode,
    run_local_hf_spec,
    run_local_hf_spec_subprocess,
    runtime_factory_from_hf_runtime,
)


def test_hf_execution_package_imports() -> None:
    assert callable(run_local_hf_spec)
    assert callable(run_local_hf_spec_subprocess)
    assert callable(build_worker_command)
    assert callable(resolve_worker_process_mode)
    assert callable(resolve_max_requests_per_worker_process)
    assert callable(is_retryable_worker_error)


def test_dispatch_module_ownership() -> None:
    assert execute_run.__module__ == "llm_conceptual_modeling.hf_execution.dispatch"
    assert (
        runtime_factory_from_hf_runtime.__module__
        == "llm_conceptual_modeling.hf_execution.dispatch"
    )
