from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis.map_extension_baseline_bundle import (
    write_map_extension_baseline_bundle,
)


def test_map_extension_bundle_resolves_remote_raw_row_paths(tmp_path: Path) -> None:
    results_root = tmp_path / "hf-map-extension-canonical"
    output_dir = tmp_path / "bundle"
    raw_row_path = (
        results_root
        / "runs"
        / "algo3"
        / "Qwen__Qwen3.5-9B"
        / "greedy"
        / "case_a"
        / "subgraph_1_to_subgraph_2"
        / "000"
        / "rep_00"
        / "raw_row.json"
    )
    raw_row_path.parent.mkdir(parents=True, exist_ok=True)
    raw_row_path.write_text(
        json.dumps(
            {
                "Source Graph": "[('a', 'b')]",
                "Target Graph": "[('x', 'y')]",
                "Mother Graph": "[('a', 'b'), ('x', 'y'), ('b', 'x')]",
                "Results": "[('b', 'bridge'), ('bridge', 'x')]",
                "Recall": 1.0,
                "model": "Qwen/Qwen3.5-9B",
                "graph_source": "case_a",
                "pair_name": "subgraph_1_to_subgraph_2",
                "decoding_algorithm": "greedy",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame.from_records(
        [
            {
                "algorithm": "algo3",
                "model": "Qwen/Qwen3.5-9B",
                "graph_source": "case_a",
                "decoding_algorithm": "greedy",
                "pair_name": "subgraph_1_to_subgraph_2",
                "replication": 0,
                "status": "finished",
                "raw_row_path": (
                    "/workspace/results/hf-open-weight-map-extension/"
                    "runs/algo3/Qwen__Qwen3.5-9B/greedy/case_a/"
                    "subgraph_1_to_subgraph_2/000/rep_00/raw_row.json"
                ),
            }
        ]
    ).to_csv(results_root / "batch_summary.csv", index=False)

    write_map_extension_baseline_bundle(
        results_root=results_root,
        output_dir=output_dir,
        random_repetitions=5,
    )

    row_level = pd.read_csv(output_dir / "row_level_baseline_comparison.csv")
    assert set(row_level["graph_source"]) == {"case_a"}
    random_rows = row_level[row_level["baseline_strategy"] == "random-k"]
    assert set(random_rows["baseline_repetition"].dropna().astype(int)) == {0, 1, 2, 3, 4}

    grouped = pd.read_csv(output_dir / "map_extension_model_vs_baseline.csv")
    assert {"graph_source", "model", "baseline_ci95_low", "baseline_ci95_high"}.issubset(
        grouped.columns
    )
