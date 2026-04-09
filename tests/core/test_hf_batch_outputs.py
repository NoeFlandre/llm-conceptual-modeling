from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.hf_batch_outputs import (
    _aggregated_analysis_spec,
    _combined_analysis_spec,
    _combined_factorial_spec,
    write_aggregated_outputs,
)


def test_write_aggregated_outputs_backfills_algo3_summary_recall_from_evaluation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_root = tmp_path / "results" / "hf-paper-batch-algo3-qwen-current"
    run_dir = (
        output_root
        / "runs"
        / "algo3"
        / "Qwen__Qwen3.5-9B"
        / "beam_num_beams_2"
        / "subgraph_1_to_subgraph_3"
        / "0000"
        / "rep_00"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    remote_raw_row_path = Path(
        "/workspace/results/hf-paper-batch-algo3-qwen-current/runs/algo3/Qwen__Qwen3.5-9B/"
        "beam_num_beams_2/subgraph_1_to_subgraph_3/0000/rep_00/raw_row.json"
    )

    raw_row = {
        "Counter-Example": "[]",
        "Depth": 2,
        "Example": "[]",
        "Mother Graph": str([("s1", "t1"), ("s2", "t2")]),
        "Number of Words": 12,
        "Recall": 0.0,
        "Repetition": 0,
        "Results": str([("s1", "t1"), ("s2", "t2")]),
        "Source Graph": str([("s1", "s2")]),
        "Source Subgraph Name": "subgraph_1",
        "Target Graph": str([("t1", "t2")]),
        "Target Subgraph Name": "subgraph_3",
        "decoding_algorithm": "beam_num_beams_2",
        "decoding_condition": "beam_num_beams_2",
        "embedding_model": "Qwen/Qwen3-Embedding-8B",
        "model": "Qwen/Qwen3.5-9B",
        "pair_name": "subgraph_1_to_subgraph_3",
        "provider": "hf-transformers",
    }
    raw_row_path = run_dir / "raw_row.json"
    raw_row_path.write_text(json.dumps(raw_row), encoding="utf-8")
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "algorithm": "algo3",
                "model": "Qwen/Qwen3.5-9B",
                "embedding_model": "Qwen/Qwen3-Embedding-8B",
                "decoding_algorithm": "beam_num_beams_2",
                "condition_label": "beam_num_beams_2",
                "pair_name": "subgraph_1_to_subgraph_3",
                "condition_bits": "0000",
                "replication": 0,
                "status": "finished",
                "thinking_mode_supported": False,
                "raw_row_path": str(remote_raw_row_path),
                "result_edge_count": 2,
                "recall": 0.0,
            }
        ),
        encoding="utf-8",
    )

    summary_frame = pd.DataFrame.from_records(
        [
            {
                "algorithm": "algo3",
                "model": "Qwen/Qwen3.5-9B",
                "embedding_model": "Qwen/Qwen3-Embedding-8B",
                "decoding_algorithm": "beam_num_beams_2",
                "condition_label": "beam_num_beams_2",
                "pair_name": "subgraph_1_to_subgraph_3",
                "condition_bits": "0000",
                "replication": 0,
                "status": "finished",
                "thinking_mode_supported": False,
                "raw_row_path": str(remote_raw_row_path),
                "result_edge_count": 2,
                "recall": 0.0,
            }
        ]
    )
    summary_frame.to_csv(output_root / "batch_summary.csv", index=False)

    monkeypatch.setattr(
        "llm_conceptual_modeling.hf_batch_outputs.run_algo3_factorial_analysis",
        lambda evaluated_path, output_path: None,
    )
    monkeypatch.setattr(
        "llm_conceptual_modeling.hf_batch_outputs.write_grouped_metric_stability",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "llm_conceptual_modeling.hf_batch_outputs.write_output_variability_analysis",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "llm_conceptual_modeling.hf_batch_outputs.write_replication_budget_analysis",
        lambda *args, **kwargs: None,
    )

    write_aggregated_outputs(output_root, summary_frame)

    evaluated_path = (
        output_root
        / "aggregated"
        / "algo3"
        / "Qwen__Qwen3.5-9B"
        / "beam_num_beams_2"
        / "evaluated.csv"
    )
    evaluated = pd.read_csv(evaluated_path)
    assert evaluated["Recall"].iloc[0] == 1.0

    updated_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert updated_summary["recall"] == 1.0
    assert updated_summary["result_edge_count"] == 2

    updated_batch_summary = pd.read_csv(output_root / "batch_summary.csv")
    assert updated_batch_summary["recall"].iloc[0] == 1.0

    raw_row_after = json.loads(raw_row_path.read_text(encoding="utf-8"))
    assert raw_row_after["Recall"] == 0.0


def test_combined_factorial_spec_matches_algorithm_shape() -> None:
    algo1_spec = _combined_factorial_spec("algo1")
    assert algo1_spec.metric_columns == ["accuracy", "recall", "precision"]
    assert algo1_spec.output_columns == ["accuracy", "recall", "precision", "Feature"]
    assert algo1_spec.factor_columns == [
        "Explanation",
        "Example",
        "Counterexample",
        "Array/List(1/-1)",
        "Tag/Adjacency(1/-1)",
        "Decoding Algorithm",
        "Beam Width Level",
        "Contrastive Penalty Level",
    ]

    algo2_spec = _combined_factorial_spec("algo2")
    assert algo2_spec.metric_columns == ["accuracy", "recall", "precision"]
    assert algo2_spec.output_columns == ["accuracy", "recall", "precision", "Feature"]
    assert algo2_spec.factor_columns == [
        "Explanation",
        "Example",
        "Counterexample",
        "Array/List(1/-1)",
        "Tag/Adjacency(1/-1)",
        "Convergence",
        "Decoding Algorithm",
        "Beam Width Level",
        "Contrastive Penalty Level",
    ]

    algo3_spec = _combined_factorial_spec("algo3")
    assert algo3_spec.metric_columns == ["Recall"]
    assert algo3_spec.output_columns == ["Recall", "Feature"]
    assert algo3_spec.factor_columns == [
        "Example",
        "Counter-Example",
        "Number of Words",
        "Depth",
        "Decoding Algorithm",
        "Beam Width Level",
        "Contrastive Penalty Level",
    ]


def test_aggregated_analysis_spec_matches_algorithm_shape() -> None:
    algo1_spec = _aggregated_analysis_spec("algo1")
    assert algo1_spec["stability_group_by"] == [
        "pair_name",
        "Explanation",
        "Example",
        "Counterexample",
        "Array/List(1/-1)",
        "Tag/Adjacency(1/-1)",
    ]
    assert algo1_spec["variability_group_by"] == [
        "pair_name",
        "Explanation",
        "Example",
        "Counterexample",
        "Array/List(1/-1)",
        "Tag/Adjacency(1/-1)",
    ]
    assert algo1_spec["metrics"] == ["accuracy", "recall", "precision"]
    assert algo1_spec["result_column"] == "Result"

    algo2_spec = _aggregated_analysis_spec("algo2")
    assert algo2_spec["stability_group_by"] == [
        "pair_name",
        "Convergence",
        "Explanation",
        "Example",
        "Counterexample",
        "Array/List(1/-1)",
        "Tag/Adjacency(1/-1)",
    ]
    assert algo2_spec["variability_group_by"] == [
        "pair_name",
        "Explanation",
        "Example",
        "Counterexample",
        "Array/List(1/-1)",
        "Tag/Adjacency(1/-1)",
        "Convergence",
    ]
    assert algo2_spec["metrics"] == ["accuracy", "recall", "precision"]
    assert algo2_spec["result_column"] == "Result"

    algo3_spec = _aggregated_analysis_spec("algo3")
    assert algo3_spec["stability_group_by"] == [
        "pair_name",
        "Depth",
        "Number of Words",
        "Example",
        "Counter-Example",
    ]
    assert algo3_spec["variability_group_by"] == [
        "pair_name",
        "Depth",
        "Number of Words",
        "Example",
        "Counter-Example",
    ]
    assert algo3_spec["metrics"] == ["Recall"]
    assert algo3_spec["result_column"] == "Results"


def test_combined_analysis_spec_matches_algorithm_shape() -> None:
    algo1_spec = _combined_analysis_spec("algo1")
    assert algo1_spec["stability_group_by"] == [
        "pair_name",
        "Explanation",
        "Example",
        "Counterexample",
        "Array/List(1/-1)",
        "Tag/Adjacency(1/-1)",
        "Decoding Algorithm",
        "Beam Width Level",
        "Contrastive Penalty Level",
    ]
    assert algo1_spec["variability_group_by"] == [
        "pair_name",
        "Explanation",
        "Example",
        "Counterexample",
        "Array/List(1/-1)",
        "Tag/Adjacency(1/-1)",
        "Decoding Algorithm",
        "Beam Width Level",
        "Contrastive Penalty Level",
    ]
    assert algo1_spec["metrics"] == ["accuracy", "recall", "precision"]
    assert algo1_spec["result_column"] == "Result"

    algo2_spec = _combined_analysis_spec("algo2")
    assert algo2_spec["stability_group_by"] == [
        "pair_name",
        "Explanation",
        "Example",
        "Counterexample",
        "Array/List(1/-1)",
        "Convergence",
        "Tag/Adjacency(1/-1)",
        "Decoding Algorithm",
        "Beam Width Level",
        "Contrastive Penalty Level",
    ]
    assert algo2_spec["variability_group_by"] == [
        "pair_name",
        "Explanation",
        "Example",
        "Counterexample",
        "Array/List(1/-1)",
        "Tag/Adjacency(1/-1)",
        "Decoding Algorithm",
        "Beam Width Level",
        "Contrastive Penalty Level",
        "Convergence",
    ]
    assert algo2_spec["metrics"] == ["accuracy", "recall", "precision"]
    assert algo2_spec["result_column"] == "Result"

    algo3_spec = _combined_analysis_spec("algo3")
    assert algo3_spec["stability_group_by"] == [
        "pair_name",
        "Depth",
        "Number of Words",
        "Example",
        "Counter-Example",
        "Decoding Algorithm",
        "Beam Width Level",
        "Contrastive Penalty Level",
    ]
    assert algo3_spec["variability_group_by"] == [
        "pair_name",
        "Depth",
        "Number of Words",
        "Example",
        "Counter-Example",
        "Decoding Algorithm",
        "Beam Width Level",
        "Contrastive Penalty Level",
    ]
    assert algo3_spec["metrics"] == ["Recall"]
    assert algo3_spec["result_column"] == "Results"
