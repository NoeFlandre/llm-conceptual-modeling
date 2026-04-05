from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.hf_batch_outputs import write_aggregated_outputs


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
