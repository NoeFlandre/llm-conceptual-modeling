from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

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


def _build_metric_overview(*, summary_path: Path, algorithm: str, factor: str) -> pd.DataFrame:
    summary = pd.read_csv(summary_path)
    level_column = factor
    records: list[dict[str, object]] = []

    for metric, metric_frame in summary.groupby("metric", dropna=False):
        levels = sorted(metric_frame[level_column].drop_duplicates().tolist())
        if len(levels) != 2:
            raise ValueError(
                f"Expected exactly two levels for {algorithm} {factor} {metric}, found {levels}"
            )
        low_level, high_level = levels

        pivot = metric_frame.pivot(index="source_input", columns=level_column, values="mean")
        low_wins = 0
        high_wins = 0
        ties = 0
        for _, row in pivot.iterrows():
            low_value = float(row[low_level])
            high_value = float(row[high_level])
            if high_value > low_value:
                high_wins += 1
            elif high_value < low_value:
                low_wins += 1
            else:
                ties += 1

        weighted_means = (
            metric_frame.assign(weighted_sum=metric_frame["mean"] * metric_frame["n"])
            .groupby(level_column, dropna=False)[["weighted_sum", "n"]]
            .sum()
        )
        low_global_mean = float(
            weighted_means.loc[low_level, "weighted_sum"] / weighted_means.loc[low_level, "n"]
        )
        high_global_mean = float(
            weighted_means.loc[high_level, "weighted_sum"] / weighted_means.loc[high_level, "n"]
        )

        records.append(
            {
                "algorithm": algorithm,
                "factor": factor,
                "metric": metric,
                "source_file_count": int(metric_frame["source_input"].nunique()),
                "level_low": _stringify_level(low_level),
                "level_high": _stringify_level(high_level),
                "global_mean_low": low_global_mean,
                "global_mean_high": high_global_mean,
                "global_mean_difference_high_minus_low": high_global_mean - low_global_mean,
                "winner_count_low": low_wins,
                "winner_count_high": high_wins,
                "tie_count": ties,
            }
        )

    return pd.DataFrame.from_records(records)


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


def _slugify(value: str) -> str:
    slug = value.lower()
    for source, target in (
        (" ", "_"),
        ("/", "_"),
        ("(", ""),
        (")", ""),
        ("-", "_"),
    ):
        slug = slug.replace(source, target)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def _stringify_level(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
