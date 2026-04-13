from __future__ import annotations

from llm_conceptual_modeling.hf_experiments import (
    _connection_metric_summary,
    _sanitize_algorithm_edge_result,
    _summary_from_raw_row,
    _trace_metric_summary,
    _validate_structural_runtime_result,
)


def test_hf_experiments_metric_helpers_are_direct_pipeline_aliases() -> None:
    assert _connection_metric_summary.__module__ == "llm_conceptual_modeling.hf_pipeline.metrics"
    assert _trace_metric_summary.__module__ == "llm_conceptual_modeling.hf_pipeline.metrics"
    assert _summary_from_raw_row.__module__ == "llm_conceptual_modeling.hf_pipeline.metrics"
    assert _validate_structural_runtime_result.__module__ == "llm_conceptual_modeling.hf_pipeline.metrics"
    assert _sanitize_algorithm_edge_result.__module__ == "llm_conceptual_modeling.hf_pipeline.metrics"
