from __future__ import annotations

from itertools import combinations
from typing import cast

import pandas as pd

from llm_conceptual_modeling.algo3.evaluation import parse_edge_list
from llm_conceptual_modeling.common.csv_schema import assert_required_columns
from llm_conceptual_modeling.common.types import Edge, PathLike


def write_output_variability_analysis(
    input_csv_paths: list[PathLike] | tuple[PathLike, ...],
    output_csv_path: PathLike,
    *,
    group_by: list[str],
    result_column: str,
) -> None:
    frames: list[pd.DataFrame] = []
    for input_csv_path in input_csv_paths:
        dataframe = pd.read_csv(input_csv_path)
        assert_required_columns(dataframe, group_by, label="group-by columns")
        assert_required_columns(dataframe, [result_column], label="result columns")

        records: list[dict[str, object]] = []
        for group_values, group_frame in dataframe.groupby(group_by, dropna=False):
            parsed_outputs = [
                _normalize_edge_set(parse_edge_list(value))
                for value in group_frame[result_column].tolist()
            ]
            group_record = {
                "source_input": str(input_csv_path),
                **_coerce_group_values(group_by, group_values),
                **_summarize_group(parsed_outputs),
            }
            records.append(group_record)

        frames.append(pd.DataFrame.from_records(records))

    output = pd.concat(frames, ignore_index=True)
    ordered_columns = [
        "source_input",
        *group_by,
        "n_outputs",
        "unique_output_count",
        "mean_edge_count",
        "sample_std_edge_count",
        "mean_pairwise_jaccard",
        "min_pairwise_jaccard",
        "exact_match_pair_rate",
        "union_edge_count",
    ]
    output = output[ordered_columns]
    output.to_csv(output_csv_path, index=False)


def _coerce_group_values(group_by: list[str], group_values: object) -> dict[str, object]:
    if len(group_by) == 1:
        if isinstance(group_values, tuple):
            return {group_by[0]: group_values[0]}
        return {group_by[0]: group_values}
    typed_group_values = cast(tuple[object, ...], group_values)
    return dict(zip(group_by, typed_group_values, strict=True))


def _normalize_edge_set(edges: list[Edge]) -> frozenset[Edge]:
    normalized_edges = {
        (str(left).strip(), str(right).strip())
        for left, right in edges
        if str(left).strip() and str(right).strip()
    }
    return frozenset(normalized_edges)


def _summarize_group(parsed_outputs: list[frozenset[Edge]]) -> dict[str, object]:
    edge_counts = pd.Series([len(output_edges) for output_edges in parsed_outputs], dtype="float64")
    pairwise_jaccards: list[float] = []
    exact_match_count = 0

    for left_edges, right_edges in combinations(parsed_outputs, 2):
        jaccard = _jaccard_similarity(left_edges, right_edges)
        pairwise_jaccards.append(jaccard)
        if left_edges == right_edges:
            exact_match_count += 1

    union_edges = frozenset().union(*parsed_outputs) if parsed_outputs else frozenset()
    pair_count = len(pairwise_jaccards)
    mean_pairwise_jaccard = 1.0 if pair_count == 0 else sum(pairwise_jaccards) / pair_count
    min_pairwise_jaccard = 1.0 if pair_count == 0 else min(pairwise_jaccards)
    exact_match_pair_rate = 1.0 if pair_count == 0 else exact_match_count / pair_count

    return {
        "n_outputs": len(parsed_outputs),
        "unique_output_count": len(set(parsed_outputs)),
        "mean_edge_count": edge_counts.mean(),
        "sample_std_edge_count": edge_counts.std(),
        "mean_pairwise_jaccard": mean_pairwise_jaccard,
        "min_pairwise_jaccard": min_pairwise_jaccard,
        "exact_match_pair_rate": exact_match_pair_rate,
        "union_edge_count": len(union_edges),
    }


def _jaccard_similarity(left_edges: frozenset[Edge], right_edges: frozenset[Edge]) -> float:
    union = left_edges | right_edges
    if not union:
        return 1.0
    intersection = left_edges & right_edges
    return len(intersection) / len(union)
