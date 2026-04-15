"""Algo3-specific pair-aware stability analysis extracted from stability_bundle.py."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis.replication_budget import write_replication_budget_analysis
from llm_conceptual_modeling.common.csv_schema import assert_required_columns


def write_algo3_pair_aware_outputs(
    *,
    results_root_path: Path,
    source_condition: Path,
    algorithm_output_dir: Path,
    manifest_records: list[dict[str, object]],
    replication_budget_overview_records: list[dict[str, object]],
) -> Algo3PairAwareOutputs | None:
    raw_input_paths = algo3_raw_input_paths(
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
    condition_frame = algo3_pair_aware_condition_frame(raw_input_paths)
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

    depth_stability_frame = algo3_level_stability_frame(condition_frame, "Depth")
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

    depth_variability_frame = algo3_level_variability_frame(condition_frame, "Depth")
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

    number_of_words_stability_frame = algo3_level_stability_frame(
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

    number_of_words_variability_frame = algo3_level_variability_frame(
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

    overall_row = algo3_metric_overview_row(condition_frame)
    variability_row = algo3_metric_variability_row(condition_frame)
    return Algo3PairAwareOutputs(
        condition_frame=condition_frame,
        overall_row=overall_row,
        variability_row=variability_row,
        depth_stability_frame=depth_stability_frame,
        depth_variability_frame=depth_variability_frame,
        number_of_words_stability_frame=number_of_words_stability_frame,
        number_of_words_variability_frame=number_of_words_variability_frame,
    )


def algo3_raw_input_paths(source_condition: Path, results_root_path: Path) -> list[Path]:
    frame = pd.read_csv(source_condition)
    if "source_input" not in frame.columns:
        return []
    raw_input_paths: list[Path] = []
    for source_input in frame["source_input"].dropna().astype(str).unique():
        resolved = resolve_source_input_path(source_input, results_root_path)
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


def algo3_pair_aware_condition_frame(raw_input_paths: list[Path]) -> pd.DataFrame:
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


def resolve_source_input_path(source_input: str, results_root_path: Path) -> Path | None:
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


def algo3_level_stability_frame(condition_frame: pd.DataFrame, level_column: str) -> pd.DataFrame:
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


def algo3_level_variability_frame(
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


def algo3_metric_overview_row(condition_frame: pd.DataFrame) -> dict[str, object]:
    metric_summary = algo3_metric_summary(condition_frame)
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


def algo3_metric_variability_row(condition_frame: pd.DataFrame) -> dict[str, object]:
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


def algo3_metric_summary(condition_frame: pd.DataFrame) -> pd.DataFrame:
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


@dataclass(frozen=True)
class Algo3PairAwareOutputs:
    condition_frame: pd.DataFrame
    overall_row: dict[str, object]
    variability_row: dict[str, object]
    depth_stability_frame: pd.DataFrame
    depth_variability_frame: pd.DataFrame
    number_of_words_stability_frame: pd.DataFrame
    number_of_words_variability_frame: pd.DataFrame
