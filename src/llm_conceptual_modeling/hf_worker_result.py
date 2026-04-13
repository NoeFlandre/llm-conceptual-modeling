"""Compatibility shim for worker result helpers."""

from llm_conceptual_modeling.hf_worker.result import (
    load_runtime_result,
    raise_missing_result_artifact,
)

__all__ = ["load_runtime_result", "raise_missing_result_artifact"]
