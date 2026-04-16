from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis._baseline_outputs import (
    write_advantage_summary,
    write_bundle_readme,
)


def test_write_advantage_summary_computes_group_stats(tmp_path: Path) -> None:
    summary_path = tmp_path / "baseline_advantage_summary.csv"
    grouped_frames = [
        pd.DataFrame(
            [
                {
                    "algorithm": "algo1",
                    "baseline_strategy": "random-k",
                    "metric": "precision",
                    "model": "m1",
                    "mean_delta": 0.25,
                },
                {
                    "algorithm": "algo1",
                    "baseline_strategy": "random-k",
                    "metric": "precision",
                    "model": "m2",
                    "mean_delta": -0.5,
                },
            ]
        )
    ]

    write_advantage_summary(grouped_frames, summary_path)

    summary = pd.read_csv(summary_path)
    assert summary.to_dict(orient="records") == [
        {
            "algorithm": "algo1",
            "baseline_strategy": "random-k",
            "metric": "precision",
            "model_count": 2,
            "models_beating_baseline": 1,
            "best_model": "m1",
            "best_model_delta": 0.25,
            "worst_model": "m2",
            "worst_model_delta": -0.5,
            "average_model_delta": -0.125,
        }
    ]


def test_write_bundle_readme_writes_expected_sections(tmp_path: Path) -> None:
    write_bundle_readme(tmp_path)

    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "# Non-LLM Baseline Comparison Bundle" in readme
    assert "`random-k`" in readme
    assert "baseline_advantage_summary.csv" in readme
