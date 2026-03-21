from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.common.csv_schema import assert_required_columns
from llm_conceptual_modeling.common.types import PathLike


def write_baseline_metric_comparison(
    input_csv_paths: list[PathLike] | tuple[PathLike, ...],
    baseline_csv_paths: list[PathLike] | tuple[PathLike, ...],
    output_csv_path: PathLike,
    *,
    metrics: list[str],
) -> None:
    baseline_means_by_name = _build_baseline_means_by_name(
        baseline_csv_paths,
        metrics=metrics,
    )
    has_single_baseline_input = len(baseline_means_by_name) == 1
    comparison_rows: list[dict[str, object]] = []

    for input_csv_path in input_csv_paths:
        file_name = Path(input_csv_path).name
        baseline_means = _resolve_baseline_means(
            baseline_means_by_name,
            file_name=file_name,
            has_single_baseline_input=has_single_baseline_input,
        )
        input_frame = pd.read_csv(input_csv_path)
        assert_required_columns(input_frame, metrics, label="metric columns")

        algorithm, model = _infer_result_metadata(input_csv_path)
        for metric in metrics:
            llm_mean = float(input_frame[metric].mean())
            baseline_mean = baseline_means[metric]
            mean_delta = llm_mean - baseline_mean
            comparison_row: dict[str, object] = {
                "algorithm": algorithm,
                "model": model,
                "metric": metric,
                "matched_file_name": file_name,
                "source_input": str(input_csv_path),
                "baseline_mean": baseline_mean,
                "llm_mean": llm_mean,
                "mean_delta": mean_delta,
            }
            comparison_rows.append(comparison_row)

    comparison_frame = pd.DataFrame(comparison_rows)
    grouped_frame = (
        comparison_frame.groupby(["algorithm", "model", "metric"], dropna=False)
        .agg(
            matched_file_count=("matched_file_name", "nunique"),
            baseline_mean=("baseline_mean", "mean"),
            llm_mean=("llm_mean", "mean"),
            mean_delta=("mean_delta", "mean"),
        )
        .reset_index()
    )
    grouped_frame.to_csv(output_csv_path, index=False)


def _resolve_baseline_means(
    baseline_means_by_name: dict[str, dict[str, float]],
    *,
    file_name: str,
    has_single_baseline_input: bool,
) -> dict[str, float]:
    if file_name in baseline_means_by_name:
        return baseline_means_by_name[file_name]
    if has_single_baseline_input:
        only_baseline_means = next(iter(baseline_means_by_name.values()))
        return only_baseline_means
    aggregate_key = _find_direct_cross_graph_key(baseline_means_by_name)
    has_aggregate_key = aggregate_key is not None
    if has_aggregate_key:
        assert aggregate_key is not None
        aggregate_baseline_means = baseline_means_by_name[aggregate_key]
        return aggregate_baseline_means
    raise ValueError(f"Missing baseline input for file name: {file_name}")


def _find_direct_cross_graph_key(
    baseline_means_by_name: dict[str, dict[str, float]],
) -> str | None:
    for baseline_file_name in baseline_means_by_name:
        contains_direct_cross_graph = "direct_cross_graph" in baseline_file_name
        if contains_direct_cross_graph:
            return baseline_file_name
    return None


def _build_baseline_means_by_name(
    baseline_csv_paths: list[PathLike] | tuple[PathLike, ...],
    *,
    metrics: list[str],
) -> dict[str, dict[str, float]]:
    baseline_means_by_name: dict[str, dict[str, float]] = {}
    for baseline_csv_path in baseline_csv_paths:
        baseline_frame = pd.read_csv(baseline_csv_path)
        assert_required_columns(baseline_frame, metrics, label="metric columns")

        file_name = Path(baseline_csv_path).name
        metric_means: dict[str, float] = {}
        for metric in metrics:
            metric_mean = float(baseline_frame[metric].mean())
            metric_means[metric] = metric_mean
        baseline_means_by_name[file_name] = metric_means
    return baseline_means_by_name


def _infer_result_metadata(input_csv_path: PathLike) -> tuple[str, str]:
    path_parts = Path(input_csv_path).parts
    for index, part in enumerate(path_parts):
        has_results_suffix = part == "results"
        has_enough_parts = index + 2 < len(path_parts)
        if has_results_suffix and has_enough_parts:
            algorithm = path_parts[index + 1]
            model = path_parts[index + 2]
            return algorithm, model
    return "unknown", "unknown"
