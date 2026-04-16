from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis._stability_helpers import _slugify
from llm_conceptual_modeling.analysis._summary_helpers import _build_metric_overview
from llm_conceptual_modeling.analysis.summary import write_grouped_metric_summary
from llm_conceptual_modeling.common.types import PathLike


@dataclass(frozen=True)
class StatisticalFactorSpec:
    column: str
    metrics: tuple[str, ...]


@dataclass(frozen=True)
class StatisticalAlgorithmSpec:
    algorithm: str
    factors: tuple[StatisticalFactorSpec, ...]


_SUMMARY_BUNDLE_SPECS: tuple[StatisticalAlgorithmSpec, ...] = (
    StatisticalAlgorithmSpec(
        algorithm="algo1",
        factors=(
            StatisticalFactorSpec("Explanation", ("accuracy", "precision", "recall")),
            StatisticalFactorSpec("Example", ("accuracy", "precision", "recall")),
            StatisticalFactorSpec("Counterexample", ("accuracy", "precision", "recall")),
            StatisticalFactorSpec("Array/List(1/-1)", ("accuracy", "precision", "recall")),
            StatisticalFactorSpec("Tag/Adjacency(1/-1)", ("accuracy", "precision", "recall")),
        ),
    ),
    StatisticalAlgorithmSpec(
        algorithm="algo2",
        factors=(
            StatisticalFactorSpec("Explanation", ("accuracy", "precision", "recall")),
            StatisticalFactorSpec("Example", ("accuracy", "precision", "recall")),
            StatisticalFactorSpec("Counterexample", ("accuracy", "precision", "recall")),
            StatisticalFactorSpec("Array/List(1/-1)", ("accuracy", "precision", "recall")),
            StatisticalFactorSpec("Tag/Adjacency(1/-1)", ("accuracy", "precision", "recall")),
            StatisticalFactorSpec("Convergence", ("accuracy", "precision", "recall")),
        ),
    ),
    StatisticalAlgorithmSpec(
        algorithm="algo3",
        factors=(
            StatisticalFactorSpec("Depth", ("Recall",)),
            StatisticalFactorSpec("Number of Words", ("Recall",)),
            StatisticalFactorSpec("Example", ("Recall",)),
            StatisticalFactorSpec("Counter-Example", ("Recall",)),
        ),
    ),
)


def write_statistical_reporting_bundle(
    *,
    results_root: PathLike,
    output_dir: PathLike,
) -> None:
    results_root_path = Path(results_root)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    manifest_records: list[dict[str, object]] = []
    overview_records: list[dict[str, object]] = []

    for algorithm_spec in _SUMMARY_BUNDLE_SPECS:
        input_glob_root = results_root_path / algorithm_spec.algorithm
        input_paths = sorted(input_glob_root.glob("*/evaluated/*.csv"))
        if not input_paths:
            raise ValueError(
                f"No evaluated CSVs found for {algorithm_spec.algorithm} under {results_root}"
            )

        algorithm_output_dir = output_dir_path / algorithm_spec.algorithm
        algorithm_output_dir.mkdir(parents=True, exist_ok=True)

        for factor_spec in algorithm_spec.factors:
            factor_slug = _slugify(factor_spec.column)
            factor_output_dir = algorithm_output_dir / factor_slug
            factor_output_dir.mkdir(parents=True, exist_ok=True)

            summary_path = factor_output_dir / "grouped_metric_summary.csv"
            metric_overview_path = factor_output_dir / "metric_overview.csv"

            write_grouped_metric_summary(
                [str(path) for path in input_paths],
                summary_path,
                group_by=[factor_spec.column],
                metrics=list(factor_spec.metrics),
            )

            metric_overview = _build_metric_overview(
                summary_path=summary_path,
                algorithm=algorithm_spec.algorithm,
                factor=factor_spec.column,
            )
            metric_overview.to_csv(metric_overview_path, index=False)

            manifest_records.append(
                {
                    "algorithm": algorithm_spec.algorithm,
                    "factor": factor_spec.column,
                    "summary_path": str(summary_path),
                    "metric_overview_path": str(metric_overview_path),
                    "source_file_count": len(input_paths),
                }
            )
            overview_records.extend(metric_overview.to_dict(orient="records"))

    pd.DataFrame.from_records(manifest_records).to_csv(
        output_dir_path / "bundle_manifest.csv",
        index=False,
    )
    pd.DataFrame.from_records(overview_records).to_csv(
        output_dir_path / "bundle_overview.csv",
        index=False,
    )
    _write_bundle_readme(output_dir_path)
def _write_bundle_readme(output_dir: Path) -> None:
    readme = """# Statistical Reporting Audit Bundle

This directory contains the organized artifacts for the statistical-reporting revision item.

## Purpose

The reviewer asked for confidence intervals and stronger statistical reporting. This bundle captures
the descriptive evidence used to answer that request across all imported evaluated result files.

## Layout

- `bundle_manifest.csv`
  Index of every generated summary file and overview file.
- `bundle_overview.csv`
  One row per algorithm, factor, and metric with global means and per-file winner counts.
- `<algorithm>/<factor_slug>/grouped_metric_summary.csv`
  The full grouped descriptive summary produced by `lcm analyze summary`.
- `<algorithm>/<factor_slug>/metric_overview.csv`
  A compact reviewer-facing overview for that factor.

## Interpretation

The grouped summary files preserve per-source-file provenance. The metric overview files compress
that evidence into global means and file-level winner counts so the revision document can cite the
most informative patterns without repeating entire CSVs inline.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
