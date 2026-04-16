from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis._variance_decomposition_spec import (
    render_variance_decomposition_table,
)


def write_variance_decomposition_outputs(
    *,
    output_dir: Path,
    decomposition: pd.DataFrame,
    algorithm_csvs: dict[str, Path],
    tables: dict[str, str],
) -> dict[str, object]:
    decomposition_csv = output_dir / "variance_decomposition.csv"
    decomposition.to_csv(decomposition_csv, index=False)

    for algorithm, algorithm_csv_path in algorithm_csvs.items():
        algorithm_frame = decomposition[decomposition["algorithm"] == algorithm].copy()
        algorithm_frame.to_csv(algorithm_csv_path, index=False)
        table = tables.get(algorithm)
        if table is None:
            table = render_variance_decomposition_table(algorithm, algorithm_frame)
        (output_dir / f"variance_decomposition_{algorithm}.tex").write_text(
            table,
            encoding="utf-8",
        )

    combined = "\n\n".join(
        tables.get(algorithm)
        or render_variance_decomposition_table(
            algorithm,
            decomposition[decomposition["algorithm"] == algorithm].copy(),
        )
        for algorithm in algorithm_csvs
    )
    combined_path = output_dir / "variance_decomposition.tex"
    combined_path.write_text(combined, encoding="utf-8")
    _write_bundle_readme(output_dir)

    return {
        "decomposition_csv": decomposition_csv,
        "algorithm_csvs": algorithm_csvs,
        "tables": tables,
        "combined_table": combined_path,
    }


def _write_bundle_readme(output_dir: Path) -> None:
    readme = """# Variance Decomposition Audit Bundle

This directory contains the organized artifacts for the variance-decomposition revision item.

## Purpose

The reviewer asked for a principled variance decomposition over the Qwen and Mistral revision
tables. This bundle captures deterministic sum-of-squares attribution per algorithm and model.

## Layout

- `variance_decomposition.csv`
  Combined decomposition across all algorithms and models.
- `variance_decomposition_<algorithm>.csv`
  Per-algorithm decomposition table.
- `variance_decomposition_<algorithm>.tex`
  Per-algorithm LaTeX table.
- `variance_decomposition.tex`
  Combined LaTeX table for all algorithms.

## Interpretation

The decomposition tables report how much of each metric's centered sum of squares is explained by
the configured factor basis and how much remains as error.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
