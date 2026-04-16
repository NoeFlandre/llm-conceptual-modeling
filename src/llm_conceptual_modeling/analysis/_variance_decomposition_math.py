from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from llm_conceptual_modeling.analysis._variance_decomposition_spec import (
    DECODING_FACTOR_NAMES,
)


def _build_term_columns(
    frame: pd.DataFrame,
    factor_order: tuple[str, ...],
) -> list[tuple[str, list[np.ndarray]]]:
    basis = {
        factor_name: _basis_columns_for_factor(frame, factor_name)
        for factor_name in factor_order
    }
    term_columns: list[tuple[str, list[np.ndarray]]] = []
    for factor_name in factor_order:
        term_columns.append((factor_name, basis[factor_name]))
    for left_name, right_name in combinations(factor_order, 2):
        if left_name in DECODING_FACTOR_NAMES and right_name in DECODING_FACTOR_NAMES:
            continue
        interaction_columns: list[np.ndarray] = []
        for left_column in basis[left_name]:
            for right_column in basis[right_name]:
                interaction_columns.append(left_column * right_column)
        term_columns.append((f"{left_name} & {right_name}", interaction_columns))
    return term_columns


def _basis_columns_for_factor(frame: pd.DataFrame, factor_name: str) -> list[np.ndarray]:
    column = frame[factor_name].astype(float).to_numpy(dtype=float)
    return [column]


def _column_sum_of_squares(column: np.ndarray, centered_response: np.ndarray) -> float:
    denominator = float(np.dot(column, column))
    if denominator == 0.0:
        return 0.0
    numerator = float(np.dot(column, centered_response))
    return (numerator**2) / denominator


def _assert_required_columns(frame: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        msg = f"Missing required variance decomposition columns: {missing}"
        raise ValueError(msg)


def _assert_balanced_cells(frame: pd.DataFrame, factor_order: tuple[str, ...]) -> None:
    counts = frame.groupby(list(factor_order), dropna=False).size()
    if counts.empty:
        msg = "Cannot decompose variance from an empty design."
        raise ValueError(msg)
    if counts.nunique() != 1:
        msg = "Variance decomposition requires a balanced design across factor cells."
        raise ValueError(msg)


def _assert_orthogonal_columns(term_columns: list[tuple[str, list[np.ndarray]]]) -> None:
    all_columns: list[np.ndarray] = []
    for _feature_name, columns in term_columns:
        all_columns.extend(columns)
    for left_index, left in enumerate(all_columns):
        for right in all_columns[left_index + 1 :]:
            dot = float(np.dot(left, right))
            if abs(dot) > 1e-8:
                msg = "Variance decomposition basis is not orthogonal."
                raise ValueError(msg)
