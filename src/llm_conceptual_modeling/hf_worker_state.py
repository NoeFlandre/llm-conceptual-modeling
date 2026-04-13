"""Compatibility shim for worker state helpers."""

from llm_conceptual_modeling.hf_worker.state import (
    mark_worker_loading_model,
    mark_worker_prefetching_model,
    mark_worker_ready_for_execution,
    read_worker_state,
    update_worker_state,
    worker_has_started_stage_execution,
)

__all__ = [
    "mark_worker_loading_model",
    "mark_worker_prefetching_model",
    "mark_worker_ready_for_execution",
    "read_worker_state",
    "update_worker_state",
    "worker_has_started_stage_execution",
]
