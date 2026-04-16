from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis._output_validity_outputs import (
    write_output_validity_bundle_outputs,
)


def test_write_output_validity_bundle_outputs_writes_bundle_files(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()

    write_output_validity_bundle_outputs(
        output_dir_path=output_dir,
        manifest_records=[
            {
                "algorithm": "all",
                "file": str(output_dir / "failure_rates.csv"),
                "description": "Failure rates",
            }
        ],
        overview_records=pd.DataFrame(
            [
                {
                    "algorithm": "algo1",
                    "model": "gpt-4o",
                    "bundle_name": "failure_rates",
                }
            ]
        ),
        failure_rates=pd.DataFrame(
            [
                {
                    "algorithm": "algo1",
                    "model": "gpt-4o",
                    "failure_rate": 0.0,
                }
            ]
        ),
        parsed_edge_counts=pd.DataFrame(
            [
                {
                    "algorithm": "algo1",
                    "model": "gpt-4o",
                    "mean": 12.0,
                }
            ]
        ),
        parsed_edge_quartiles=pd.DataFrame(
            [
                {
                    "algorithm": "algo1",
                    "model": "gpt-4o",
                    "median": 10.0,
                }
            ]
        ),
    )

    assert (output_dir / "bundle_manifest.csv").exists()
    assert (output_dir / "bundle_overview.csv").exists()
    assert (output_dir / "failure_rates.csv").exists()
    assert (output_dir / "parsed_edge_counts.csv").exists()
    assert (output_dir / "parsed_edge_quartiles.csv").exists()
    assert (output_dir / "README.md").read_text(encoding="utf-8").startswith(
        "# Output Validity and Breadth Audit Bundle"
    )
