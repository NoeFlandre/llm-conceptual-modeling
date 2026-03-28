from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis.stability_bundle import write_stability_bundle


def test_write_stability_bundle_adds_replication_budget_outputs(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir(parents=True)
    output_dir = tmp_path / "bundle"

    _write_flat_file(
        results_root / "overall_metric_stability_by_algorithm.csv",
        [
            "algorithm,metric,condition_count,mean_cv,median_cv,max_cv,mean_range_width,max_range_width",
            "algo1,accuracy,2,0.01,0.0,0.02,0.03,0.04",
        ],
    )
    _write_flat_file(
        results_root / "algo1_condition_stability.csv",
        [
            "source_input,Explanation,metric,n,mean,sample_std,min,max,range_width,coefficient_of_variation",
            "algo1/file1.csv,-1,accuracy,5,100.0,12.0,90.0,110.0,20.0,0.12",
            "algo1/file2.csv,1,accuracy,5,50.0,0.0,50.0,50.0,0.0,0.0",
        ],
    )

    write_stability_bundle(results_root=results_root, output_dir=output_dir)

    detailed = pd.read_csv(output_dir / "algo1" / "replication_budget_by_condition.csv")
    assert "required_total_runs" in detailed.columns
    assert "additional_runs_needed" in detailed.columns
    assert detailed["required_total_runs"].tolist() == [23, 5]
    assert detailed["additional_runs_needed"].tolist() == [18, 0]

    overview = pd.read_csv(output_dir / "replication_budget_overview.csv")
    assert set(overview.columns) >= {
        "algorithm",
        "metric",
        "condition_count",
        "max_required_total_runs",
        "max_additional_runs_needed",
        "conditions_needing_more_runs",
    }
    row = overview.iloc[0]
    assert row["algorithm"] == "algo1"
    assert row["metric"] == "accuracy"
    assert row["condition_count"] == 2
    assert row["max_required_total_runs"] == 23
    assert row["max_additional_runs_needed"] == 18
    assert row["conditions_needing_more_runs"] == 1


def _write_flat_file(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
