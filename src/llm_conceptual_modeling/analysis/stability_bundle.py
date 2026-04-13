from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis.replication_budget import write_replication_budget_analysis
from llm_conceptual_modeling.common.csv_schema import assert_required_columns
from llm_conceptual_modeling.common.types import PathLike


@dataclass(frozen=True)
class StabilityAlgorithmSpec:
    algorithm: str
    level_factors: tuple[str, ...]


@dataclass(frozen=True)
class Algo3PairAwareOutputs:
    condition_frame: pd.DataFrame
    overall_row: dict[str, object]
    variability_row: dict[str, object]
    depth_stability_frame: pd.DataFrame
    depth_variability_frame: pd.DataFrame
    number_of_words_stability_frame: pd.DataFrame
    number_of_words_variability_frame: pd.DataFrame


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
                algo3_pair_aware_outputs = _write_algo3_pair_aware_outputs(
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
                for _, row in (
                    budget_frame.groupby("metric", dropna=False)
                    .agg(
                        condition_count=("metric", "size"),
                        max_required_total_runs=("required_total_runs", "max"),
                        max_additional_runs_needed=("additional_runs_needed", "max"),
                        mean_required_total_runs=("required_total_runs", "mean"),
                        conditions_needing_more_runs=(
                            "additional_runs_needed",
                            lambda series: int((series > 0).sum()),
                        ),
                    )
                    .reset_index()
                    .iterrows()
                ):
                    replication_budget_overview_records.append(
                        {
                            "algorithm": algorithm_spec.algorithm,
                            "metric": row["metric"],
                            "condition_count": int(row["condition_count"]),
                            "max_required_total_runs": int(row["max_required_total_runs"]),
                            "max_additional_runs_needed": int(row["max_additional_runs_needed"]),
                            "mean_required_total_runs": float(row["mean_required_total_runs"]),
                            "conditions_needing_more_runs": int(
                                row["conditions_needing_more_runs"]
                            ),
                        }
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


def _write_algo3_pair_aware_outputs(
    *,
    results_root_path: Path,
    source_condition: Path,
    algorithm_output_dir: Path,
    manifest_records: list[dict[str, object]],
    replication_budget_overview_records: list[dict[str, object]],
) -> Algo3PairAwareOutputs | None:
    raw_input_paths = _algo3_raw_input_paths(
        source_condition=source_condition,
        results_root_path=results_root_path,
    )
    if not raw_input_paths:
        dest = algorithm_output_dir / "condition_stability.csv"
        pd.read_csv(source_condition).to_csv(dest, index=False)
        manifest_records.append(
            {
                "algorithm": "algo3",
                "factor": "condition",
                "relative_path": "algo3/condition_stability.csv",
                "description": (
                    "Per-file, per-condition stability statistics for ALGO3 across the five "
                    "repetitions. Fallback pooled layout preserved because pair-aware raw "
                    "evaluated files were not available."
                ),
            }
        )
        replication_budget_path = algorithm_output_dir / "replication_budget_by_condition.csv"
        write_replication_budget_analysis([source_condition], replication_budget_path)
        manifest_records.append(
            {
                "algorithm": "algo3",
                "factor": "condition",
                "relative_path": "algo3/replication_budget_by_condition.csv",
                "description": (
                    "Per-condition replication budget for ALGO3 using a 95% CI z-score of "
                    "1.96 and a 5% relative half-width target. Key columns: "
                    "required_total_runs, additional_runs_needed."
                ),
            }
        )
        return None

    condition_path = algorithm_output_dir / "condition_stability.csv"
    condition_frame = _algo3_pair_aware_condition_frame(raw_input_paths)
    condition_frame.to_csv(condition_path, index=False)
    manifest_records.append(
        {
            "algorithm": "algo3",
            "factor": "condition",
            "relative_path": "algo3/condition_stability.csv",
            "description": (
                "Pair-aware per-condition stability statistics for ALGO3 across the five "
                "repetitions. Conditions are counted at the LLM × pair × factorial-condition "
                "level, matching the algo1/algo2 reporting scale."
            ),
        }
    )

    replication_budget_path = algorithm_output_dir / "replication_budget_by_condition.csv"
    write_replication_budget_analysis([condition_path], replication_budget_path)
    manifest_records.append(
        {
            "algorithm": "algo3",
            "factor": "condition",
            "relative_path": "algo3/replication_budget_by_condition.csv",
            "description": (
                "Per-condition replication budget for ALGO3 using a 95% CI z-score of "
                "1.96 and a 5% relative half-width target. Key columns: "
                "required_total_runs, additional_runs_needed."
            ),
        }
    )
    budget_frame = pd.read_csv(replication_budget_path)
    for _, row in (
        budget_frame.groupby("metric", dropna=False)
        .agg(
            condition_count=("metric", "size"),
            max_required_total_runs=("required_total_runs", "max"),
            max_additional_runs_needed=("additional_runs_needed", "max"),
            mean_required_total_runs=("required_total_runs", "mean"),
            conditions_needing_more_runs=(
                "additional_runs_needed",
                lambda series: int((series > 0).sum()),
            ),
        )
        .reset_index()
        .iterrows()
    ):
        replication_budget_overview_records.append(
            {
                "algorithm": "algo3",
                "metric": row["metric"],
                "condition_count": int(row["condition_count"]),
                "max_required_total_runs": int(row["max_required_total_runs"]),
                "max_additional_runs_needed": int(row["max_additional_runs_needed"]),
                "mean_required_total_runs": float(row["mean_required_total_runs"]),
                "conditions_needing_more_runs": int(row["conditions_needing_more_runs"]),
            }
        )

    depth_stability_frame = _algo3_level_stability_frame(condition_frame, "Depth")
    depth_stability_frame.to_csv(algorithm_output_dir / "depth_stability_by_level.csv", index=False)
    manifest_records.append(
        {
            "algorithm": "algo3",
            "factor": "Depth",
            "relative_path": "algo3/depth_stability_by_level.csv",
            "description": (
                "Pair-aware ALGO3 stability summaries grouped by Depth. Columns: n, mean_cv, "
                "median_cv, mean_range_width, max_range_width."
            ),
        }
    )

    depth_variability_frame = _algo3_level_variability_frame(condition_frame, "Depth")
    depth_variability_frame.to_csv(
        algorithm_output_dir / "depth_variability_incidence.csv",
        index=False,
    )
    manifest_records.append(
        {
            "algorithm": "algo3",
            "factor": "Depth",
            "relative_path": "algo3/depth_variability_incidence.csv",
            "description": (
                "Pair-aware ALGO3 incidence of any run-to-run variation grouped by Depth. "
                "Key column: varying_condition_share."
            ),
        }
    )

    number_of_words_stability_frame = _algo3_level_stability_frame(
        condition_frame,
        "Number of Words",
    )
    number_of_words_stability_frame.to_csv(
        algorithm_output_dir / "number_of_words_stability_by_level.csv",
        index=False,
    )
    manifest_records.append(
        {
            "algorithm": "algo3",
            "factor": "Number of Words",
            "relative_path": "algo3/number_of_words_stability_by_level.csv",
            "description": (
                "Pair-aware ALGO3 stability summaries grouped by Number of Words. "
                "Columns: n, mean_cv, median_cv, mean_range_width, max_range_width."
            ),
        }
    )

    number_of_words_variability_frame = _algo3_level_variability_frame(
        condition_frame,
        "Number of Words",
    )
    number_of_words_variability_frame.to_csv(
        algorithm_output_dir / "number_of_words_variability_incidence.csv",
        index=False,
    )
    manifest_records.append(
        {
            "algorithm": "algo3",
            "factor": "Number of Words",
            "relative_path": "algo3/number_of_words_variability_incidence.csv",
            "description": (
                "Pair-aware ALGO3 incidence of any run-to-run variation grouped by Number of "
                "Words. Key column: varying_condition_share."
            ),
        }
    )

    overall_row = _algo3_metric_overview_row(condition_frame)
    variability_row = _algo3_metric_variability_row(condition_frame)
    return Algo3PairAwareOutputs(
        condition_frame=condition_frame,
        overall_row=overall_row,
        variability_row=variability_row,
        depth_stability_frame=depth_stability_frame,
        depth_variability_frame=depth_variability_frame,
        number_of_words_stability_frame=number_of_words_stability_frame,
        number_of_words_variability_frame=number_of_words_variability_frame,
    )


def _algo3_raw_input_paths(source_condition: Path, results_root_path: Path) -> list[Path]:
    frame = pd.read_csv(source_condition)
    if "source_input" not in frame.columns:
        return []
    raw_input_paths: list[Path] = []
    for source_input in frame["source_input"].dropna().astype(str).unique():
        resolved = _resolve_source_input_path(source_input, results_root_path)
        if resolved is None:
            return []
        if resolved not in raw_input_paths:
            raw_input_paths.append(resolved)
    if not raw_input_paths:
        return []
    for raw_input_path in raw_input_paths:
        raw_frame = pd.read_csv(raw_input_path)
        if "pair_name" not in raw_frame.columns and not {
            "Source Subgraph Name",
            "Target Subgraph Name",
        }.issubset(raw_frame.columns):
            return []
    return raw_input_paths


def _algo3_pair_aware_condition_frame(raw_input_paths: list[Path]) -> pd.DataFrame:
    stability_frames: list[pd.DataFrame] = []
    for raw_input_path in raw_input_paths:
        raw_frame = pd.read_csv(raw_input_path)
        if "pair_name" not in raw_frame.columns:
            raw_frame = raw_frame.copy()
            raw_frame["pair_name"] = (
                raw_frame["Source Subgraph Name"].astype(str)
                + "_to_"
                + raw_frame["Target Subgraph Name"].astype(str)
            )
        assert_required_columns(
            raw_frame,
            ["pair_name", "Depth", "Number of Words", "Example", "Counter-Example", "Recall"],
            label="algo3 pair-aware stability columns",
        )
        stability = (
            raw_frame.groupby(
                ["pair_name", "Depth", "Number of Words", "Example", "Counter-Example"],
                dropna=False,
            )["Recall"]
            .agg(["count", "mean", "std", "min", "max"])
            .reset_index()
            .rename(columns={"count": "n", "std": "sample_std"})
        )
        stability["range_width"] = stability["max"] - stability["min"]
        stability["coefficient_of_variation"] = (
            stability["sample_std"] / stability["mean"]
        )
        stability["metric"] = "Recall"
        stability["source_input"] = str(raw_input_path)
        stability_frames.append(stability)
    output = pd.concat(stability_frames, ignore_index=True)
    ordered_columns = [
        "source_input",
        "pair_name",
        "Depth",
        "Number of Words",
        "Example",
        "Counter-Example",
        "metric",
        "n",
        "mean",
        "sample_std",
        "min",
        "max",
        "range_width",
        "coefficient_of_variation",
    ]
    return output[ordered_columns]


def _resolve_source_input_path(source_input: str, results_root_path: Path) -> Path | None:
    source_path = Path(source_input)
    if source_path.exists():
        return source_path
    if not source_path.is_absolute():
        candidate = results_root_path / source_path
        if candidate.exists():
            return candidate
    source_text = str(source_path)
    workspace_prefix = "/workspace/results/"
    if source_text.startswith(workspace_prefix):
        candidate = results_root_path / source_text.removeprefix(workspace_prefix)
        if candidate.exists():
            return candidate
    if workspace_prefix in source_text:
        candidate = results_root_path / source_text.split(workspace_prefix, maxsplit=1)[1]
        if candidate.exists():
            return candidate
    return None


def _algo3_level_stability_frame(condition_frame: pd.DataFrame, level_column: str) -> pd.DataFrame:
    stability = (
        condition_frame.groupby([level_column, "metric"], dropna=False)[
            "coefficient_of_variation"
        ]
        .agg(["count", "mean", "median"])
        .reset_index()
        .rename(
            columns={
                "count": "condition_count",
                "mean": "mean_cv",
                "median": "median_cv",
            }
        )
    )
    range_width = (
        condition_frame.groupby([level_column, "metric"], dropna=False)["range_width"]
        .agg(["mean", "max"])
        .reset_index()
        .rename(columns={"mean": "mean_range_width", "max": "max_range_width"})
    )
    merged = stability.merge(range_width, on=[level_column, "metric"], how="left")
    return merged[
        [
            level_column,
            "metric",
            "condition_count",
            "mean_cv",
            "median_cv",
            "mean_range_width",
            "max_range_width",
        ]
    ]


def _algo3_level_variability_frame(
    condition_frame: pd.DataFrame,
    level_column: str,
) -> pd.DataFrame:
    variable_frame = condition_frame.copy()
    variable_frame["_varying_condition"] = variable_frame["sample_std"].fillna(0).ne(0)
    summary = (
        variable_frame.groupby([level_column, "metric"], dropna=False)
        .agg(
            condition_count=("metric", "size"),
            varying_condition_count=("_varying_condition", "sum"),
        )
        .reset_index()
    )
    summary["varying_condition_share"] = (
        summary["varying_condition_count"] / summary["condition_count"]
    )
    summary["varying_condition_count"] = summary["varying_condition_count"].astype(int)
    return summary[
        [
            level_column,
            "metric",
            "condition_count",
            "varying_condition_count",
            "varying_condition_share",
        ]
    ]


def _algo3_metric_overview_row(condition_frame: pd.DataFrame) -> dict[str, object]:
    metric_summary = _algo3_metric_summary(condition_frame)
    if len(metric_summary) != 1:
        raise ValueError("Expected exactly one algo3 metric summary row.")
    row = metric_summary.iloc[0]
    return {
        "algorithm": "algo3",
        "metric": row["metric"],
        "condition_count": int(row["condition_count"]),
        "mean_cv": float(row["mean_cv"]),
        "median_cv": float(row["median_cv"]),
        "max_cv": float(row["max_cv"]),
        "mean_range_width": float(row["mean_range_width"]),
        "max_range_width": float(row["max_range_width"]),
    }


def _algo3_metric_variability_row(condition_frame: pd.DataFrame) -> dict[str, object]:
    variable_frame = condition_frame.copy()
    variable_frame["_varying_condition"] = variable_frame["sample_std"].fillna(0).ne(0)
    summary = (
        variable_frame.groupby("metric", dropna=False)
        .agg(
            condition_count=("metric", "size"),
            varying_condition_count=("_varying_condition", "sum"),
        )
        .reset_index()
    )
    if len(summary) != 1:
        raise ValueError("Expected exactly one algo3 variability summary row.")
    row = summary.iloc[0]
    return {
        "algorithm": "algo3",
        "metric": row["metric"],
        "condition_count": int(row["condition_count"]),
        "varying_condition_count": int(row["varying_condition_count"]),
        "varying_condition_share": float(
            row["varying_condition_count"] / row["condition_count"]
        ),
    }


def _algo3_metric_summary(condition_frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        condition_frame.groupby("metric", dropna=False)
        .agg(
            condition_count=("metric", "size"),
            mean_cv=("coefficient_of_variation", "mean"),
            median_cv=("coefficient_of_variation", "median"),
            max_cv=("coefficient_of_variation", "max"),
            mean_range_width=("range_width", "mean"),
            max_range_width=("range_width", "max"),
        )
        .reset_index()
    )
    return summary


def _frame_to_overview_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        records.append(
            {
                "algorithm": row["algorithm"],
                "metric": row["metric"],
                "condition_count": int(row["condition_count"]),
                "mean_cv": float(row["mean_cv"]),
                "median_cv": float(row["median_cv"]),
                "max_cv": float(row["max_cv"]),
                "mean_range_width": float(row["mean_range_width"]),
                "max_range_width": float(row["max_range_width"]),
            }
        )
    return records


def _patch_algorithm_rows(
    frame: pd.DataFrame,
    replacement_row: dict[str, object],
) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "algorithm" not in frame.columns or "metric" not in frame.columns:
        return frame
    algorithm = replacement_row["algorithm"]
    metric = replacement_row["metric"]
    filtered = frame.loc[~((frame["algorithm"] == algorithm) & (frame["metric"] == metric))].copy()
    replacement_frame = pd.DataFrame.from_records([replacement_row])
    return pd.concat([filtered, replacement_frame], ignore_index=True)


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
