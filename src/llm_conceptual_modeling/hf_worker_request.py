"""Compatibility shim for worker request helpers."""

from llm_conceptual_modeling.hf_worker.request import (
    HFWorkerRequest,
    enqueue_worker_request,
    load_worker_request,
)

__all__ = ["HFWorkerRequest", "enqueue_worker_request", "load_worker_request"]
