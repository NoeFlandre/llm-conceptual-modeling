from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis._algo3_stability import (
    Algo3PairAwareOutputs as Algo3PairAwareOutputs,
)
from llm_conceptual_modeling.analysis._algo3_stability import (
    write_algo3_pair_aware_outputs,
)
from llm_conceptual_modeling.analysis._stability_budget import (
    build_replication_budget_overview_records,
)
from llm_conceptual_modeling.analysis._stability_helpers import _slugify
from llm_conceptual_modeling.analysis._stability_outputs import (
    write_stability_bundle_outputs,
)
from llm_conceptual_modeling.analysis.replication_budget import write_replication_budget_analysis
from llm_conceptual_modeling.common.types import PathLike


@dataclass(frozen=True)
class StabilityAlgorithmSpec:
    algorithm: str
    level_factors: tuple[str, ...]


_STABILITY_BUNDLE_SPECS: tuple[StabilityAlgorithmSpec, ...] = (
    StabilityAlgorithmSpec(
        algorithm="algo1",
        level_factors=(
            "Explanation",
            "Example",
            "Counterexample",
            "Array/List(1/-1)",
            "Tag/Adjacency(1/-1)",
        ),
    ),
    StabilityAlgorithmSpec(
        algorithm="algo2",
        level_factors=(
            "Convergence",
            "Explanation",
            "Example",
            "Counterexample",
            "Array/List(1/-1)",
            "Tag/Adjacency(1/-1)",
        ),
    ),
    StabilityAlgorithmSpec(
        algorithm="algo3",
        level_factors=(
            "Depth",
            "Number of Words",
            "Example",
            "Counter-Example",
        ),
    ),
)


def write_stability_bundle(
    *,
    results_root: PathLike,
    output_dir: PathLike,
) -> None:
    results_root_path = Path(results_root)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    manifest_records: list[dict[str, object]] = []
    overview_records: list[dict[str, object]] = []
    source_variability_frame: pd.DataFrame | None = None
    source_overall_frame: pd.DataFrame | None = None
    algo3_pair_aware_outputs: Algo3PairAwareOutputs | None = None

    # --- top-level cross-algorithm summaries ---
    source_variability = results_root_path / "variability_incidence_by_algorithm.csv"
    source_overall = results_root_path / "overall_metric_stability_by_algorithm.csv"

    if source_variability.exists():
        source_variability_frame = pd.read_csv(source_variability)
        manifest_records.append(
            {
                "algorithm": "cross_algorithm",
                "factor": "overall",
                "relative_path": "variability_incidence_by_algorithm.csv",
                "description": (
                    "Cross-algorithm count and share of conditions that changed "
                    "across the five repetitions. Key column: varying_condition_share."
                ),
            }
        )
    if source_overall.exists():
        source_overall_frame = pd.read_csv(source_overall)
        manifest_records.append(
            {
                "algorithm": "cross_algorithm",
                "factor": "overall",
                "relative_path": "overall_metric_stability_by_algorithm.csv",
                "description": (
                    "Cross-algorithm coefficient of variation and range-width summaries "
                    "by metric. Key columns: mean_cv, median_cv, mean_range_width."
                ),
            }
        )

    replication_budget_overview_records: list[dict[str, object]] = []

    # --- per-algorithm subdirectories ---
    for algorithm_spec in _STABILITY_BUNDLE_SPECS:
        algorithm_output_dir = output_dir_path / algorithm_spec.algorithm
        algorithm_output_dir.mkdir(parents=True, exist_ok=True)

        # Condition-stability file (exists as algo*_condition_stability.csv flat)
        source_condition = (
            results_root_path / f"{algorithm_spec.algorithm}_condition_stability.csv"
        )
        if source_condition.exists():
            if algorithm_spec.algorithm == "algo3":
                algo3_pair_aware_outputs = write_algo3_pair_aware_outputs(
                    results_root_path=results_root_path,
                    source_condition=source_condition,
                    algorithm_output_dir=algorithm_output_dir,
                    manifest_records=manifest_records,
                    replication_budget_overview_records=replication_budget_overview_records,
                )
            else:
                dest = algorithm_output_dir / "condition_stability.csv"
                pd.read_csv(source_condition).to_csv(dest, index=False)
                manifest_records.append(
                    {
                        "algorithm": algorithm_spec.algorithm,
                        "factor": "condition",
                        "relative_path": f"{algorithm_spec.algorithm}/condition_stability.csv",
                        "description": (
                            f"Per-file, per-condition stability statistics for "
                            f"{algorithm_spec.algorithm.upper()} across the five repetitions. "
                            "Columns: n, mean, sample_std, min, max, "
                            "range_width, coefficient_of_variation."
                        ),
                    }
                )

                replication_budget_path = (
                    algorithm_output_dir / "replication_budget_by_condition.csv"
                )
                write_replication_budget_analysis(
                    [source_condition],
                    replication_budget_path,
                )
                manifest_records.append(
                    {
                        "algorithm": algorithm_spec.algorithm,
                        "factor": "condition",
                        "relative_path": (
                            f"{algorithm_spec.algorithm}/replication_budget_by_condition.csv"
                        ),
                        "description": (
                            "Per-condition replication budget for "
                            f"{algorithm_spec.algorithm.upper()} "
                            "using a 95% CI z-score of 1.96 and a 5% relative half-width target. "
                            "Key columns: required_total_runs, additional_runs_needed."
                        ),
                    }
                )
                budget_frame = pd.read_csv(replication_budget_path)
                replication_budget_overview_records.extend(
                    build_replication_budget_overview_records(
                        algorithm=algorithm_spec.algorithm,
                        budget_frame=budget_frame,
                    )
                )

        # Level-specific stability and variability-incidence files
        for level_factor in algorithm_spec.level_factors:
            factor_slug = _slugify(level_factor)
            if (
                algorithm_spec.algorithm == "algo3"
                and algo3_pair_aware_outputs is not None
            ):
                if level_factor == "Depth":
                    level_stability_frame = algo3_pair_aware_outputs.depth_stability_frame
                    level_incidence_frame = algo3_pair_aware_outputs.depth_variability_frame
                elif level_factor == "Number of Words":
                    level_stability_frame = (
                        algo3_pair_aware_outputs.number_of_words_stability_frame
                    )
                    level_incidence_frame = (
                        algo3_pair_aware_outputs.number_of_words_variability_frame
                    )
                else:
                    level_stability_frame = None
                    level_incidence_frame = None
            else:
                level_stability_frame = None
                level_incidence_frame = None

            source_stability = (
                results_root_path
                / f"{algorithm_spec.algorithm}_{factor_slug}_stability_by_level.csv"
            )
            if level_stability_frame is not None:
                dest = algorithm_output_dir / f"{factor_slug}_stability_by_level.csv"
                level_stability_frame.to_csv(dest, index=False)
                manifest_records.append(
                    {
                        "algorithm": algorithm_spec.algorithm,
                        "factor": level_factor,
                        "relative_path": (
                            f"{algorithm_spec.algorithm}/{factor_slug}_stability_by_level.csv"
                        ),
                        "description": (
                            f"Aggregated {algorithm_spec.algorithm.upper()} stability "
                            f"summaries grouped by {level_factor!r}. "
                            "Columns: n, mean_cv, median_cv, "
                            "mean_range_width, max_range_width."
                        ),
                    }
                )
            elif source_stability.exists():
                dest = algorithm_output_dir / f"{factor_slug}_stability_by_level.csv"
                pd.read_csv(source_stability).to_csv(dest, index=False)
                manifest_records.append(
                    {
                        "algorithm": algorithm_spec.algorithm,
                        "factor": level_factor,
                        "relative_path": (
                            f"{algorithm_spec.algorithm}/{factor_slug}_stability_by_level.csv"
                        ),
                        "description": (
                            f"Aggregated {algorithm_spec.algorithm.upper()} stability "
                            f"summaries grouped by {level_factor!r}. "
                            "Columns: n, mean_cv, median_cv, "
                            "mean_range_width, max_range_width."
                        ),
                    }
                )

            source_incidence = (
                results_root_path
                / f"{algorithm_spec.algorithm}_{factor_slug}_variability_incidence.csv"
            )
            if level_incidence_frame is not None:
                dest = algorithm_output_dir / f"{factor_slug}_variability_incidence.csv"
                level_incidence_frame.to_csv(dest, index=False)
                manifest_records.append(
                    {
                        "algorithm": algorithm_spec.algorithm,
                        "factor": level_factor,
                        "relative_path": (
                            f"{algorithm_spec.algorithm}/{factor_slug}_variability_incidence.csv"
                        ),
                        "description": (
                            f"Incidence of any run-to-run variation in "
                            f"{algorithm_spec.algorithm.upper()} "
                            f"grouped by {level_factor!r}. "
                            "Key column: varying_condition_share."
                        ),
                    }
                )
            elif source_incidence.exists():
                dest = algorithm_output_dir / f"{factor_slug}_variability_incidence.csv"
                pd.read_csv(source_incidence).to_csv(dest, index=False)
                manifest_records.append(
                    {
                        "algorithm": algorithm_spec.algorithm,
                        "factor": level_factor,
                        "relative_path": (
                            f"{algorithm_spec.algorithm}/{factor_slug}_variability_incidence.csv"
                        ),
                        "description": (
                            f"Incidence of any run-to-run variation in "
                            f"{algorithm_spec.algorithm.upper()} "
                            f"grouped by {level_factor!r}. "
                            "Key column: varying_condition_share."
                        ),
                    }
                )

    write_stability_bundle_outputs(
        output_dir_path=output_dir_path,
        manifest_records=manifest_records,
        overview_records=overview_records,
        replication_budget_overview_records=replication_budget_overview_records,
        source_variability_frame=source_variability_frame,
        source_overall_frame=source_overall_frame,
        algo3_pair_aware_outputs=algo3_pair_aware_outputs,
    )
