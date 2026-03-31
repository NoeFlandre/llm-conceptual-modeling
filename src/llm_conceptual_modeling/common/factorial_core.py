import csv
import itertools

import numpy as np
import pandas as pd

from llm_conceptual_modeling.common.csv_schema import assert_required_columns
from llm_conceptual_modeling.common.types import (
    GeneralizedFactorialSpec,
    MultiMetricFactorialSpec,
    PathLike,
)


def run_multi_metric_factorial_analysis(
    input_csv_paths: list[PathLike] | tuple[PathLike, ...],
    output_path: PathLike,
    spec: MultiMetricFactorialSpec,
) -> None:
    all_results: dict[str, dict[str, list[float]]] = {}
    for input_csv_path in input_csv_paths:
        results = _process_single_csv(input_csv_path, spec)
        for metric_name, features in results.items():
            if metric_name not in all_results:
                all_results[metric_name] = {}
            for feature_name, value in features.items():
                all_results[metric_name].setdefault(feature_name, []).append(value)

    all_features = set()
    for metric_name in all_results:
        all_features.update(all_results[metric_name].keys())

    def sort_key(feature: str) -> tuple[int, str]:
        return (1, feature) if "_AND_" not in feature else (2, feature)

    with open(str(output_path), "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(spec.output_columns)
        for feature in sorted(all_features, key=sort_key):
            metric_values: list[float] = [
                sum(all_results[metric].get(feature, [0.0]))
                / len(all_results[metric].get(feature, [1.0]))
                if feature in all_results[metric]
                else 0.0
                for metric in spec.metric_columns
            ]
            writer.writerow([*metric_values, feature])


def _process_single_csv(
    csv_file: PathLike,
    spec: MultiMetricFactorialSpec,
) -> dict[str, dict[str, float]]:
    results_by_metric, num_experiments = _create_factorial_design(csv_file, spec)
    normalization_factor = 2**num_experiments
    processed_results: dict[str, dict[str, float]] = {}

    for metric_name, dataframe in results_by_metric.items():
        column_sums = [
            (column, dataframe[column].sum() / normalization_factor)
            for column in dataframe.columns
        ]

        squared_values: list[tuple[str, float]] = []
        for index, (name, value) in enumerate(column_sums):
            squared_values.append(
                (name, value if index == 0 else (value**2) * normalization_factor)
            )

        total_sum_of_squares = sum(value for _, value in squared_values[1:])
        processed_results[metric_name] = {}
        for name, value in squared_values[1:]:
            normalized = (value / total_sum_of_squares) * 100 if total_sum_of_squares != 0 else 0.0
            processed_results[metric_name][_remove_evaluating_suffix(name)] = normalized

    return processed_results


def _create_factorial_design(
    csv_file: PathLike,
    spec: MultiMetricFactorialSpec,
) -> tuple[dict[str, pd.DataFrame], int]:
    dataframes_by_metric, num_experiments = _read_metric_csv(csv_file, spec)
    results: dict[str, pd.DataFrame] = {}

    for metric_name, dataframe in dataframes_by_metric.items():
        factor_frame = dataframe[spec.factor_columns]
        metric_columns = dataframe.columns.difference(spec.factor_columns)

        for factor_column in factor_frame.columns:
            for metric_column in metric_columns:
                dataframe[f"{factor_column}_evaluating_{metric_column}"] = (
                    dataframe[factor_column] * dataframe[metric_column]
                )

        for combination in itertools.combinations(factor_frame.columns, 2):
            interaction_name = "_AND_".join(combination)
            dataframe[interaction_name] = dataframe[list(combination)].prod(axis=1)
            for metric_column in metric_columns:
                dataframe[f"{interaction_name}_evaluating_{metric_column}"] = (
                    dataframe[interaction_name] * dataframe[metric_column]
                )

        kept_columns = [
            column
            for column in dataframe.columns
            if "_evaluating_" in column or column in metric_columns
        ]
        results[metric_name] = dataframe[kept_columns]

    return results, num_experiments


def _read_metric_csv(
    csv_file: PathLike,
    spec: MultiMetricFactorialSpec,
) -> tuple[dict[str, pd.DataFrame], int]:
    dataframe = pd.read_csv(csv_file, sep=",", encoding="utf-8")
    if len(dataframe.columns) <= 1:
        msg = f"File {csv_file} appears not to be comma-delimited."
        raise ValueError(msg)

    assert_required_columns(dataframe, spec.factor_columns, label="factor columns")
    assert_required_columns(dataframe, spec.metric_columns, label="metric columns")

    processed = dataframe[spec.factor_columns + spec.metric_columns].copy()
    processed.dropna(subset=spec.metric_columns, inplace=True)

    by_metric: dict[str, pd.DataFrame] = {}
    for metric_column in spec.metric_columns:
        by_metric[metric_column] = pd.concat(
            [processed[spec.factor_columns], processed[[metric_column]]],
            axis=1,
        )

    return by_metric, len(spec.factor_columns)


def _remove_evaluating_suffix(value: str) -> str:
    position = value.find("_evaluating")
    return value[:position] if position != -1 else value


def run_generalized_factorial_analysis(
    input_csv_paths: list[PathLike] | tuple[PathLike, ...],
    output_path: PathLike,
    spec: GeneralizedFactorialSpec,
) -> None:
    frame = pd.concat(
        [pd.read_csv(path, sep=",", encoding="utf-8") for path in input_csv_paths],
        ignore_index=True,
    )
    required_columns = list(spec.factor_columns) + list(spec.metric_columns)
    if spec.replication_column is not None:
        required_columns.append(spec.replication_column)
    assert_required_columns(frame, required_columns, label="generalized factorial columns")
    working = frame[required_columns].copy()
    working.dropna(subset=spec.metric_columns, inplace=True)

    term_builders: list[tuple[str, list[str]]] = []
    term_frames: dict[str, pd.DataFrame] = {}
    factor_term_builders: list[tuple[str, list[str]]] = []
    for factor_column in spec.factor_columns:
        encoded = _encode_factor_column(working, factor_column)
        term_frames[factor_column] = encoded
        encoded_columns = list(encoded.columns)
        term_builders.append((factor_column, encoded_columns))
        factor_term_builders.append((factor_column, encoded_columns))

    if spec.replication_column is not None:
        encoded = _encode_factor_column(working, spec.replication_column)
        term_frames[spec.replication_column] = encoded
        term_builders.append((spec.replication_column, list(encoded.columns)))

    if spec.include_pairwise_interactions:
        for (left_name, left_columns), (right_name, right_columns) in itertools.combinations(
            factor_term_builders, 2
        ):
            interaction_name = f"{left_name}_AND_{right_name}"
            interaction_columns: dict[str, pd.Series] = {}
            left_frame = term_frames[left_name]
            right_frame = term_frames[right_name]
            for left_column in left_columns:
                for right_column in right_columns:
                    column_name = (
                        f"{interaction_name}__{left_column}__{right_column}"
                    )
                    interaction_columns[column_name] = (
                        left_frame[left_column] * right_frame[right_column]
                    )
            interaction_frame = pd.DataFrame(interaction_columns)
            term_frames[interaction_name] = interaction_frame
            term_builders.append((interaction_name, list(interaction_frame.columns)))

    intercept = pd.DataFrame({"Intercept": np.ones(len(working), dtype=float)})
    design_frames = [intercept]
    for term_name, _column_names in term_builders:
        design_frames.append(term_frames[term_name])
    full_design = pd.concat(design_frames, axis=1)
    full_matrix = full_design.to_numpy(dtype=float)

    rows: list[list[object]] = []
    for term_name, column_names in term_builders:
        metric_values: list[float] = []
        for metric_name in spec.metric_columns:
            metric_series = working[metric_name].astype(float)
            total_ss = float(((metric_series - metric_series.mean()) ** 2).sum())
            if total_ss == 0.0:
                metric_values.append(0.0)
                continue
            response = metric_series.to_numpy(dtype=float)
            full_sse = _fit_sse(full_matrix, response)
            reduced_design = full_design.drop(columns=column_names)
            reduced_sse = _fit_sse(reduced_design.to_numpy(dtype=float), response)
            ss_effect = max(0.0, reduced_sse - full_sse)
            metric_values.append((ss_effect / total_ss) * 100)
        rows.append([*metric_values, term_name])

    if spec.replication_column is not None:
        error_values: list[float] = []
        for metric_name in spec.metric_columns:
            metric_series = working[metric_name].astype(float)
            total_ss = float(((metric_series - metric_series.mean()) ** 2).sum())
            if total_ss == 0.0:
                error_values.append(0.0)
                continue
            response = metric_series.to_numpy(dtype=float)
            full_sse = _fit_sse(full_matrix, response)
            error_values.append((full_sse / total_ss) * 100)
        rows.append([*error_values, "Error"])

    with open(str(output_path), "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(spec.output_columns)
        for row in rows:
            writer.writerow(row)


def _encode_factor_column(frame: pd.DataFrame, column_name: str) -> pd.DataFrame:
    series = frame[column_name]
    if _is_binary_series(series) or _is_sparse_level_series(series):
        return pd.DataFrame({column_name: series.astype(float)})
    encoded = pd.get_dummies(series.astype(str), prefix=column_name)
    if encoded.shape[1] <= 1:
        return pd.DataFrame({column_name: np.zeros(len(series), dtype=float)})
    baseline_column = sorted(encoded.columns)[0]
    return encoded.drop(columns=[baseline_column]).astype(float)


def _fit_sse(design_matrix: np.ndarray, response: np.ndarray) -> float:
    coefficients, *_ = np.linalg.lstsq(design_matrix, response, rcond=None)
    fitted = design_matrix @ coefficients
    residuals = response - fitted
    return float(np.square(residuals).sum())


def _is_binary_series(series: pd.Series) -> bool:
    try:
        unique_values = {float(value) for value in series.dropna().unique()}
    except (TypeError, ValueError):
        return False
    return unique_values.issubset({-1.0, 1.0}) and bool(unique_values)


def _is_sparse_level_series(series: pd.Series) -> bool:
    try:
        unique_values = {float(value) for value in series.dropna().unique()}
    except (TypeError, ValueError):
        return False
    return unique_values.issubset({-1.0, 0.0, 1.0}) and bool(unique_values)
