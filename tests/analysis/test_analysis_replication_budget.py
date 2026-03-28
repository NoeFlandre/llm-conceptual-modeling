from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.cli import main


def test_cli_analyze_replication_budget_writes_required_run_counts(tmp_path: Path) -> None:
    input_path = tmp_path / "condition_stability.csv"
    output_path = tmp_path / "replication_budget.csv"
    pd.DataFrame(
        {
            "source_input": ["a.csv", "b.csv", "c.csv"],
            "Condition": ["A", "B", "C"],
            "metric": ["accuracy", "accuracy", "recall"],
            "n": [5, 5, 5],
            "mean": [100.0, 50.0, 0.0],
            "sample_std": [12.0, 0.0, 0.0],
        }
    ).to_csv(input_path, index=False)

    exit_code = main(
        [
            "analyze",
            "replication-budget",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0

    actual = pd.read_csv(output_path)

    first_row = actual.iloc[0]
    assert first_row["required_total_runs"] == 23
    assert first_row["additional_runs_needed"] == 18
    assert first_row["precision_margin"] == 5.0
    assert first_row["relative_half_width_target"] == 0.05
    assert first_row["z_score"] == 1.96
    assert first_row["requirement_status"] == "needs_more_runs"

    zero_std_row = actual.iloc[1]
    assert zero_std_row["required_total_runs"] == 5
    assert zero_std_row["additional_runs_needed"] == 0
    assert zero_std_row["requirement_status"] == "satisfied"

    zero_mean_zero_std_row = actual.iloc[2]
    assert zero_mean_zero_std_row["required_total_runs"] == 5
    assert zero_mean_zero_std_row["additional_runs_needed"] == 0
    assert zero_mean_zero_std_row["precision_margin"] == 0.0
    assert zero_mean_zero_std_row["requirement_status"] == "satisfied_zero_mean"
