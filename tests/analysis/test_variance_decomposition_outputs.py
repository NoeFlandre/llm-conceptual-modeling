from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis._variance_decomposition_outputs import (
    write_variance_decomposition_outputs,
)


def test_write_variance_decomposition_outputs_writes_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "variance"
    output_dir.mkdir()

    decomposition = pd.DataFrame(
        [
            {
                "algorithm": "algo1",
                "model": "gpt-5",
                "feature": "Main",
                "metric": "accuracy",
                "ss": 1.0,
                "pct_with_error": 50.0,
                "pct_without_error": 50.0,
            }
        ]
    )
    tables = {"algo1": "\\begin{tabular}{l}"}
    algorithm_csvs = {"algo1": output_dir / "variance_decomposition_algo1.csv"}

    result = write_variance_decomposition_outputs(
        output_dir=output_dir,
        decomposition=decomposition,
        algorithm_csvs=algorithm_csvs,
        tables=tables,
    )

    assert (output_dir / "variance_decomposition.csv").exists()
    assert (output_dir / "variance_decomposition_algo1.csv").exists()
    assert (output_dir / "variance_decomposition_algo1.tex").exists()
    assert (output_dir / "variance_decomposition.tex").exists()
    assert (output_dir / "README.md").read_text(encoding="utf-8").startswith(
        "# Variance Decomposition Audit Bundle"
    )
    assert result["decomposition_csv"] == output_dir / "variance_decomposition.csv"
