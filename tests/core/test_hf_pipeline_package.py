from llm_conceptual_modeling.hf_pipeline.algorithms import run_algo1, run_algo2, run_algo3
from llm_conceptual_modeling.hf_pipeline.metrics import (
    connection_metric_summary,
    sanitize_algorithm_edge_result,
    summary_from_raw_row,
    trace_metric_summary,
    validate_structural_runtime_result,
)


def test_hf_pipeline_package_imports() -> None:
    assert callable(run_algo1)
    assert callable(run_algo2)
    assert callable(run_algo3)
    assert callable(connection_metric_summary)
    assert callable(trace_metric_summary)
    assert callable(summary_from_raw_row)
    assert callable(validate_structural_runtime_result)
    assert callable(sanitize_algorithm_edge_result)
