from llm_conceptual_modeling.hf_worker import main, serve_request_queue
from llm_conceptual_modeling.hf_worker.policy import (
    resolve_run_retry_attempts,
    resolve_stage_timeout_seconds,
    resolve_startup_timeout_seconds,
)
from llm_conceptual_modeling.hf_worker.request import (
    enqueue_worker_request,
    load_worker_request,
)
from llm_conceptual_modeling.hf_worker.result import load_runtime_result
from llm_conceptual_modeling.hf_worker.state import (
    mark_worker_prefetching_model,
    mark_worker_ready_for_execution,
    update_worker_state,
)


def test_hf_worker_helpers_live_in_the_worker_package() -> None:
    assert load_runtime_result.__module__ == "llm_conceptual_modeling.hf_worker.result"
    assert enqueue_worker_request.__module__ == "llm_conceptual_modeling.hf_worker.request"
    assert load_worker_request.__module__ == "llm_conceptual_modeling.hf_worker.request"
    assert mark_worker_prefetching_model.__module__ == "llm_conceptual_modeling.hf_worker.state"
    assert mark_worker_ready_for_execution.__module__ == "llm_conceptual_modeling.hf_worker.state"
    assert update_worker_state.__module__ == "llm_conceptual_modeling.hf_worker.state"
    assert resolve_startup_timeout_seconds.__module__ == "llm_conceptual_modeling.hf_worker.policy"
    assert resolve_stage_timeout_seconds.__module__ == "llm_conceptual_modeling.hf_worker.policy"
    assert resolve_run_retry_attempts.__module__ == "llm_conceptual_modeling.hf_worker.policy"
    assert main.__module__ == "llm_conceptual_modeling.hf_worker.entrypoint"
    assert serve_request_queue.__module__ == "llm_conceptual_modeling.hf_worker.entrypoint"
