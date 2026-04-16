from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis._variability_outputs import (
    write_variability_bundle_outputs,
)


def test_write_variability_bundle_outputs_writes_bundle_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()

    write_variability_bundle_outputs(
        output_dir_path=output_dir,
        manifest_records=[
            {
                "relative_path": "algorithm_output_variability_summary.csv",
                "description": "Cross-algorithm variability summary",
            }
        ],
        overview_records=[
            {
                "algorithm": "algo1",
                "mean_pairwise_jaccard": 1.0,
            }
        ],
    )

    assert (output_dir / "bundle_manifest.csv").exists()
    assert (output_dir / "bundle_overview.csv").exists()
    assert (output_dir / "README.md").read_text(encoding="utf-8").startswith(
        "# Raw Output Variability Audit Bundle"
    )
    overview = pd.read_csv(output_dir / "bundle_overview.csv")
    assert overview.iloc[0]["algorithm"] == "algo1"
