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
from llm_conceptual_modeling.analysis._stability_helpers import (
    _slugify,
)
from llm_conceptual_modeling.analysis._stability_helpers import (
    frame_to_overview_records as _frame_to_overview_records,
)
from llm_conceptual_modeling.analysis._stability_helpers import (
    patch_algorithm_rows as _patch_algorithm_rows,
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
    variability_incidence_path = output_dir_path / "variability_incidence_by_algorithm.csv"
    overall_stability_path = output_dir_path / "overall_metric_stability_by_algorithm.csv"

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

    if source_overall_frame is not None:
        patched_overall_frame = source_overall_frame.copy()
        if algo3_pair_aware_outputs is not None:
            patched_overall_frame = _patch_algorithm_rows(
                patched_overall_frame,
                algo3_pair_aware_outputs.overall_row,
            )
        patched_overall_frame.to_csv(overall_stability_path, index=False)
        overview_records.extend(
            _frame_to_overview_records(patched_overall_frame)
        )

    if source_variability_frame is not None:
        patched_variability_frame = source_variability_frame.copy()
        if algo3_pair_aware_outputs is not None:
            patched_variability_frame = _patch_algorithm_rows(
                patched_variability_frame,
                algo3_pair_aware_outputs.variability_row,
            )
        patched_variability_frame.to_csv(variability_incidence_path, index=False)

    # --- write bundle metadata ---
    pd.DataFrame.from_records(manifest_records).to_csv(
        output_dir_path / "bundle_manifest.csv",
        index=False,
    )
    if overview_records:
        pd.DataFrame.from_records(overview_records).to_csv(
            output_dir_path / "bundle_overview.csv",
            index=False,
        )
    if replication_budget_overview_records:
        pd.DataFrame.from_records(replication_budget_overview_records).to_csv(
            output_dir_path / "replication_budget_overview.csv",
            index=False,
        )
        manifest_records.append(
            {
                "algorithm": "cross_algorithm",
                "factor": "overall",
                "relative_path": "replication_budget_overview.csv",
                "description": (
                    "Conservative required-run summary by algorithm and metric under the "
                    "95% CI precision calculation. Key columns: max_required_total_runs, "
                    "max_additional_runs_needed, conditions_needing_more_runs."
                ),
            }
        )
        pd.DataFrame.from_records(manifest_records).to_csv(
            output_dir_path / "bundle_manifest.csv",
            index=False,
        )
    _write_bundle_readme(output_dir_path)


def _write_bundle_readme(output_dir: Path) -> None:
    readme = """# Replication Stability Audit Bundle

This directory contains the organized artifacts for the replication-stability revision item.

## Purpose

The reviewer asked for a principled justification of the five-replication decision. This bundle
captures repetition-level stability analysis over the five recorded runs in the imported corpus.
It now also adds a conservative confidence-interval run calculation using

  n = ((1.96 * s) / (r * |x_bar|))^2

with a 95% CI (`1.96`) and a 5% relative half-width target (`r = 0.05`).

## Layout

- `bundle_manifest.csv`
  Index of every generated file with descriptions.
- `bundle_overview.csv`
  Cross-algorithm stability summary (CV and range-width) by algorithm and metric.
- `variability_incidence_by_algorithm.csv`
  Cross-algorithm count and share of conditions that changed across repetitions.
- `overall_metric_stability_by_algorithm.csv`
  Coefficient-of-variation and range-width summaries by algorithm and metric.
- `replication_budget_overview.csv`
  Conservative required-run summary by algorithm and metric under the CI precision rule.
- `<algorithm>/condition_stability.csv`
  Per-file, per-condition stability statistics across the five repetitions.
- `<algorithm>/replication_budget_by_condition.csv`
  Per-condition required total runs and additional runs needed under the CI precision rule.
- `<algorithm>/<factor>_stability_by_level.csv`
  Aggregated stability summaries grouped by a specific factor level.
- `<algorithm>/<factor>_variability_incidence.csv`
  Incidence of any run-to-run variation grouped by a specific factor level.

## Key Interpretation

- **ALGO1 and ALGO2 are nearly repetition-stable**: median CVs are 0.0 across most conditions.
- **ALGO2 is especially stable when Convergence = 1**: zero varying conditions.
- **ALGO3 is orders of magnitude noisier**: median CV on recall is 3.87, meaning noise is
  nearly 4 times the signal size — not a small or marginal difference.
- **The CI-based run budget is conservative**: low-variance conditions can be justified with the
  existing runs, while low-mean high-variance conditions can demand many more.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
