from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis.hypothesis import write_paired_factor_hypothesis_tests
from llm_conceptual_modeling.common.types import PathLike


@dataclass(frozen=True)
class HypothesisFactorSpec:
    factor: str
    metrics: tuple[str, ...]
    pair_by: tuple[str, ...]


@dataclass(frozen=True)
class HypothesisAlgorithmSpec:
    algorithm: str
    factors: tuple[HypothesisFactorSpec, ...]


_HYPOTHESIS_BUNDLE_SPECS: tuple[HypothesisAlgorithmSpec, ...] = (
    HypothesisAlgorithmSpec(
        algorithm="algo1",
        factors=(
            HypothesisFactorSpec(
                factor="Explanation",
                metrics=("accuracy", "precision", "recall"),
                pair_by=(
                    "Repetition",
                    "Example",
                    "Counterexample",
                    "Array/List(1/-1)",
                    "Tag/Adjacency(1/-1)",
                ),
            ),
            HypothesisFactorSpec(
                factor="Example",
                metrics=("accuracy", "precision", "recall"),
                pair_by=(
                    "Repetition",
                    "Explanation",
                    "Counterexample",
                    "Array/List(1/-1)",
                    "Tag/Adjacency(1/-1)",
                ),
            ),
            HypothesisFactorSpec(
                factor="Counterexample",
                metrics=("accuracy", "precision", "recall"),
                pair_by=(
                    "Repetition",
                    "Explanation",
                    "Example",
                    "Array/List(1/-1)",
                    "Tag/Adjacency(1/-1)",
                ),
            ),
            HypothesisFactorSpec(
                factor="Array/List(1/-1)",
                metrics=("accuracy", "precision", "recall"),
                pair_by=(
                    "Repetition",
                    "Explanation",
                    "Example",
                    "Counterexample",
                    "Tag/Adjacency(1/-1)",
                ),
            ),
            HypothesisFactorSpec(
                factor="Tag/Adjacency(1/-1)",
                metrics=("accuracy", "precision", "recall"),
                pair_by=(
                    "Repetition",
                    "Explanation",
                    "Example",
                    "Counterexample",
                    "Array/List(1/-1)",
                ),
            ),
        ),
    ),
    HypothesisAlgorithmSpec(
        algorithm="algo2",
        factors=(
            HypothesisFactorSpec(
                factor="Convergence",
                metrics=("accuracy", "precision", "recall"),
                pair_by=(
                    "Repetition",
                    "Explanation",
                    "Example",
                    "Counterexample",
                    "Array/List(1/-1)",
                    "Tag/Adjacency(1/-1)",
                ),
            ),
            HypothesisFactorSpec(
                factor="Explanation",
                metrics=("accuracy", "precision", "recall"),
                pair_by=(
                    "Repetition",
                    "Example",
                    "Counterexample",
                    "Array/List(1/-1)",
                    "Tag/Adjacency(1/-1)",
                    "Convergence",
                ),
            ),
            HypothesisFactorSpec(
                factor="Example",
                metrics=("accuracy", "precision", "recall"),
                pair_by=(
                    "Repetition",
                    "Explanation",
                    "Counterexample",
                    "Array/List(1/-1)",
                    "Tag/Adjacency(1/-1)",
                    "Convergence",
                ),
            ),
            HypothesisFactorSpec(
                factor="Counterexample",
                metrics=("accuracy", "precision", "recall"),
                pair_by=(
                    "Repetition",
                    "Explanation",
                    "Example",
                    "Array/List(1/-1)",
                    "Tag/Adjacency(1/-1)",
                    "Convergence",
                ),
            ),
            HypothesisFactorSpec(
                factor="Array/List(1/-1)",
                metrics=("accuracy", "precision", "recall"),
                pair_by=(
                    "Repetition",
                    "Explanation",
                    "Example",
                    "Counterexample",
                    "Tag/Adjacency(1/-1)",
                    "Convergence",
                ),
            ),
            HypothesisFactorSpec(
                factor="Tag/Adjacency(1/-1)",
                metrics=("accuracy", "precision", "recall"),
                pair_by=(
                    "Repetition",
                    "Explanation",
                    "Example",
                    "Counterexample",
                    "Array/List(1/-1)",
                    "Convergence",
                ),
            ),
        ),
    ),
    HypothesisAlgorithmSpec(
        algorithm="algo3",
        factors=(
            HypothesisFactorSpec(
                factor="Depth",
                metrics=("Recall",),
                pair_by=(
                    "Repetition",
                    "Example",
                    "Counter-Example",
                    "Number of Words",
                    "Source Subgraph Name",
                    "Target Subgraph Name",
                ),
            ),
            HypothesisFactorSpec(
                factor="Number of Words",
                metrics=("Recall",),
                pair_by=(
                    "Repetition",
                    "Example",
                    "Counter-Example",
                    "Depth",
                    "Source Subgraph Name",
                    "Target Subgraph Name",
                ),
            ),
            HypothesisFactorSpec(
                factor="Example",
                metrics=("Recall",),
                pair_by=(
                    "Repetition",
                    "Counter-Example",
                    "Depth",
                    "Number of Words",
                    "Source Subgraph Name",
                    "Target Subgraph Name",
                ),
            ),
            HypothesisFactorSpec(
                factor="Counter-Example",
                metrics=("Recall",),
                pair_by=(
                    "Repetition",
                    "Example",
                    "Depth",
                    "Number of Words",
                    "Source Subgraph Name",
                    "Target Subgraph Name",
                ),
            ),
        ),
    ),
)


def write_hypothesis_testing_bundle(
    *,
    results_root: PathLike,
    output_dir: PathLike,
) -> None:
    results_root_path = Path(results_root)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    manifest_records: list[dict[str, object]] = []
    overview_records: list[dict[str, object]] = []

    for algorithm_spec in _HYPOTHESIS_BUNDLE_SPECS:
        input_paths = sorted(
            (results_root_path / algorithm_spec.algorithm).glob("*/evaluated/*.csv")
        )
        if not input_paths:
            raise ValueError(
                f"No evaluated CSVs found for {algorithm_spec.algorithm} under {results_root}"
            )

        algorithm_output_dir = output_dir_path / algorithm_spec.algorithm
        algorithm_output_dir.mkdir(parents=True, exist_ok=True)

        for factor_spec in algorithm_spec.factors:
            factor_slug = _slugify(factor_spec.factor)
            factor_output_dir = algorithm_output_dir / factor_slug
            factor_output_dir.mkdir(parents=True, exist_ok=True)

            paired_tests_path = factor_output_dir / "paired_tests.csv"
            significance_summary_path = factor_output_dir / "significance_summary.csv"
            factor_overview_path = factor_output_dir / "factor_overview.csv"

            write_paired_factor_hypothesis_tests(
                [str(path) for path in input_paths],
                paired_tests_path,
                factor=factor_spec.factor,
                pair_by=list(factor_spec.pair_by),
                metrics=list(factor_spec.metrics),
            )

            paired_tests = pd.read_csv(paired_tests_path)
            significance_summary = _build_significance_summary(paired_tests)
            significance_summary.to_csv(significance_summary_path, index=False)

            factor_overview = _build_factor_overview(
                paired_tests=paired_tests,
                algorithm=algorithm_spec.algorithm,
                factor=factor_spec.factor,
            )
            factor_overview.to_csv(factor_overview_path, index=False)

            manifest_records.append(
                {
                    "algorithm": algorithm_spec.algorithm,
                    "factor": factor_spec.factor,
                    "paired_tests_path": str(paired_tests_path),
                    "significance_summary_path": str(significance_summary_path),
                    "factor_overview_path": str(factor_overview_path),
                    "source_file_count": len(input_paths),
                }
            )
            overview_records.extend(factor_overview.to_dict(orient="records"))

    pd.DataFrame.from_records(manifest_records).to_csv(
        output_dir_path / "bundle_manifest.csv",
        index=False,
    )
    pd.DataFrame.from_records(overview_records).to_csv(
        output_dir_path / "bundle_overview.csv",
        index=False,
    )
    _write_bundle_readme(output_dir_path)


def _build_significance_summary(paired_tests: pd.DataFrame) -> pd.DataFrame:
    summary = paired_tests.copy()
    summary["is_significant"] = summary["p_value_adjusted"] <= 0.05
    summary["direction"] = summary.apply(_direction_label, axis=1)
    grouped_counts = summary.groupby(
        ["metric", "direction", "is_significant"],
        dropna=False,
    ).size()
    return grouped_counts.to_frame("test_count").reset_index()


def _build_factor_overview(
    *,
    paired_tests: pd.DataFrame,
    algorithm: str,
    factor: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for metric, metric_frame in paired_tests.groupby("metric", dropna=False):
        significant = metric_frame[metric_frame["p_value_adjusted"] <= 0.05]
        if significant.empty:
            strongest_row = metric_frame.sort_values(
                by=["p_value_adjusted", "p_value", "source_input"],
                na_position="last",
            ).iloc[0]
            strongest_source = str(strongest_row["source_input"])
            strongest_adjusted_p = strongest_row["p_value_adjusted"]
            strongest_effect_size = strongest_row["effect_size_paired_d"]
        else:
            strongest_row = significant.reindex(
                significant["effect_size_paired_d"].abs().sort_values(ascending=False).index
            ).iloc[0]
            strongest_source = str(strongest_row["source_input"])
            strongest_adjusted_p = strongest_row["p_value_adjusted"]
            strongest_effect_size = strongest_row["effect_size_paired_d"]

        records.append(
            {
                "algorithm": algorithm,
                "factor": factor,
                "metric": metric,
                "test_count": int(len(metric_frame)),
                "significant_test_count": int((metric_frame["p_value_adjusted"] <= 0.05).sum()),
                "significant_share": float((metric_frame["p_value_adjusted"] <= 0.05).mean()),
                "mean_difference_average": float(metric_frame["mean_difference"].mean()),
                "mean_absolute_effect_size": float(
                    metric_frame["effect_size_paired_d"].abs().mean()
                ),
                "positive_direction_count": int((metric_frame["mean_difference"] > 0).sum()),
                "negative_direction_count": int((metric_frame["mean_difference"] < 0).sum()),
                "null_direction_count": int((metric_frame["mean_difference"] == 0).sum()),
                "strongest_source_input": strongest_source,
                "strongest_adjusted_p_value": strongest_adjusted_p,
                "strongest_effect_size_paired_d": strongest_effect_size,
            }
        )
    return pd.DataFrame.from_records(records)


def _direction_label(row: pd.Series) -> str:
    mean_difference = row["mean_difference"]
    if pd.isna(mean_difference) or mean_difference == 0:
        return "equal"
    return "high_gt_low" if mean_difference > 0 else "high_lt_low"


def _write_bundle_readme(output_dir: Path) -> None:
    readme = """# Hypothesis-Testing Audit Bundle

This directory contains the organized artifacts for the formal hypothesis-testing revision item.

## Purpose

The reviewer asked for p-values, multiple-comparison adjustment, and stronger inferential support.
This bundle captures those paired tests in a structure that can be read by factor rather than as a
flat directory of unrelated CSV files.

## Layout

- `bundle_manifest.csv`
  Index of every generated paired-test file, significance-summary file, and factor overview file.
- `bundle_overview.csv`
  One row per algorithm, factor, and metric with significant-test counts, direction counts, and
  the strongest adjusted result.
- `<algorithm>/<factor_slug>/paired_tests.csv`
  Full paired t-test output for that factor.
- `<algorithm>/<factor_slug>/significance_summary.csv`
  Count summary by metric, direction, and adjusted-significance status.
- `<algorithm>/<factor_slug>/factor_overview.csv`
  Compact reviewer-facing overview for that factor.
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
