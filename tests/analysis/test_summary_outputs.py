from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis._summary_outputs import (
    write_summary_bundle_outputs,
)


def test_write_summary_bundle_outputs_writes_bundle_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()

    write_summary_bundle_outputs(
        output_dir_path=output_dir,
        manifest_records=[
            {
                "algorithm": "algo1",
                "factor": "Explanation",
                "summary_path": "algo1/explanation/grouped_metric_summary.csv",
                "metric_overview_path": "algo1/explanation/metric_overview.csv",
                "source_file_count": 1,
            }
        ],
        overview_records=[
            {
                "algorithm": "algo1",
                "factor": "Explanation",
                "metric": "accuracy",
                "mean_value": 0.5,
            }
        ],
    )

    assert (output_dir / "bundle_manifest.csv").exists()
    assert (output_dir / "bundle_overview.csv").exists()
    assert (output_dir / "README.md").read_text(encoding="utf-8").startswith(
        "# Statistical Reporting Audit Bundle"
    )
    overview = pd.read_csv(output_dir / "bundle_overview.csv")
    assert overview.iloc[0]["metric"] == "accuracy"
