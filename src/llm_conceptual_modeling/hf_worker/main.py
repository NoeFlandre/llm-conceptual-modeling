"""Compatibility shim for the canonical worker entrypoint module."""

from llm_conceptual_modeling.hf_worker.entrypoint import main, serve_request_queue

__all__ = ["main", "serve_request_queue"]
