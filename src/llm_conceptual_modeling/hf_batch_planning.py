"""Compatibility shim for batch-planning helpers."""

from llm_conceptual_modeling.hf_batch.planning import (
    default_runtime_profile_provider,
    plan_paper_batch_specs,
)

__all__ = ["default_runtime_profile_provider", "plan_paper_batch_specs"]
