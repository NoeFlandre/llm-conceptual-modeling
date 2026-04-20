import ast
import csv
import itertools

import pandas as pd

from llm_conceptual_modeling.common.types import PathLike

FACTOR_COLUMNS = ["Example", "Counter-Example", "Number of Words", "Depth"]


def _parse_result_to_metric(result: str) -> int | float:
    if result == "Empty":
        return 0
    try:
        parsed = ast.literal_eval(result)
    except (ValueError, SyntaxError):
        return float("nan")

    if isinstance(parsed, list):
        return len(parsed)
    if isinstance(parsed, tuple):
        return 1
    return float("nan")


def _read_recall_csv(csv_file: PathLike) -> tuple[pd.DataFrame, int]:
    dataframe = pd.read_csv(csv_file, sep=",", encoding="utf-8")

    required_columns = FACTOR_COLUMNS + ["Recall"]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        msg = f"Missing required columns: {missing_columns}"
        raise ValueError(msg)

    if "Results" in dataframe.columns:
        dataframe["NumEdges"] = dataframe["Results"].apply(_parse_result_to_metric)
    elif "Result" in dataframe.columns:
        dataframe["NumEdges"] = dataframe["Result"].apply(_parse_result_to_metric)

    processed = dataframe[required_columns].copy()
    processed.dropna(subset=["Recall"], inplace=True)
    processed["Number of Words"] = processed["Number of Words"].map({3: -1, 5: 1})
    processed["Depth"] = processed["Depth"].map({1: -1, 2: 1})
    return processed, len(FACTOR_COLUMNS)


def run_factorial_analysis(input_csv_path: PathLike, output_path: PathLike) -> None:
    dataframe, num_experiments = _read_recall_csv(input_csv_path)
    metric_column = "Recall"
    normalization_factor = 2**num_experiments

    working = dataframe[FACTOR_COLUMNS + [metric_column]].copy()
    factor_frame = working[FACTOR_COLUMNS]

    for factor_column in factor_frame.columns:
        working[f"{factor_column}_evaluating_{metric_column}"] = (
            working[factor_column] * working[metric_column]
        )

    for combination in itertools.combinations(factor_frame.columns, 2):
        interaction_name = "_AND_".join(combination)
        working[interaction_name] = working[list(combination)].prod(axis=1)
        working[f"{interaction_name}_evaluating_{metric_column}"] = (
            working[interaction_name] * working[metric_column]
        )

    kept_columns = [
        column for column in working.columns if "_evaluating_" in column or column == metric_column
    ]
    evaluated = working[kept_columns]

    column_sums: list[tuple[str, float]] = []
    for column in evaluated.columns:
        column_sums.append((column, evaluated[column].sum() / normalization_factor))

    squared_values: list[tuple[str, float]] = []
    for index, (name, value) in enumerate(column_sums):
        if index == 0:
            squared_values.append((name, value))
        else:
            squared_values.append((name, (value**2) * normalization_factor))

    total_sum_of_squares = sum(value for _, value in squared_values[1:])

    with open(str(output_path), "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Feature", "Result"])
        for name, value in squared_values[1:]:
            feature_name = _remove_evaluating_suffix(name)
            normalized = (value / total_sum_of_squares) * 100 if total_sum_of_squares != 0 else 0.0
            if float(normalized).is_integer():
                normalized = int(normalized)
            writer.writerow([feature_name, normalized])


def _remove_evaluating_suffix(value: str) -> str:
    position = value.find("_evaluating")
    return value[:position] if position != -1 else value
