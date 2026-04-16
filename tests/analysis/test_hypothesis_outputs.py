from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis._hypothesis_outputs import (
    write_hypothesis_bundle_outputs,
)


def test_write_hypothesis_bundle_outputs_writes_manifest_overview_and_readme(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()

    write_hypothesis_bundle_outputs(
        output_dir_path=output_dir,
        manifest_records=[
            {
                "algorithm": "algo1",
                "factor": "Explanation",
                "paired_tests_path": "algo1/explanation/paired_tests.csv",
                "significance_summary_path": "algo1/explanation/significance_summary.csv",
                "factor_overview_path": "algo1/explanation/factor_overview.csv",
                "source_file_count": 1,
            }
        ],
        overview_records=[
            {
                "algorithm": "algo1",
                "factor": "Explanation",
                "metric": "accuracy",
                "significant_test_count": 1,
            }
        ],
    )

    assert (output_dir / "bundle_manifest.csv").exists()
    assert (output_dir / "bundle_overview.csv").exists()
    assert (output_dir / "README.md").read_text(encoding="utf-8").startswith(
        "# Hypothesis-Testing Audit Bundle"
    )
    overview = pd.read_csv(output_dir / "bundle_overview.csv")
    assert overview.iloc[0]["factor"] == "Explanation"
