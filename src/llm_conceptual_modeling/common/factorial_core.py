import csv
import itertools

import pandas as pd

from llm_conceptual_modeling.common.csv_schema import assert_required_columns
from llm_conceptual_modeling.common.types import MultiMetricFactorialSpec, PathLike


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
