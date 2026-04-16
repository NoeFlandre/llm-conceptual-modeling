from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llm_conceptual_modeling.analysis._stability_helpers import _slugify
from llm_conceptual_modeling.analysis._summary_helpers import _build_metric_overview
from llm_conceptual_modeling.analysis._summary_outputs import (
    write_summary_bundle_outputs,
)
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

    write_summary_bundle_outputs(
        output_dir_path=output_dir_path,
        manifest_records=manifest_records,
        overview_records=overview_records,
    )
