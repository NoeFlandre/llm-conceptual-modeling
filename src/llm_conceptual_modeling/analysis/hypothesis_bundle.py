from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis._hypothesis_bundle_helpers import (
    _build_factor_overview,
    _build_significance_summary,
)
from llm_conceptual_modeling.analysis._hypothesis_outputs import (
    write_hypothesis_bundle_outputs,
)
from llm_conceptual_modeling.analysis._stability_helpers import _slugify
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

    write_hypothesis_bundle_outputs(
        output_dir_path=output_dir_path,
        manifest_records=manifest_records,
        overview_records=overview_records,
    )
