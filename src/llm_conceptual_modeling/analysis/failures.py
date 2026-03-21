import pandas as pd

from llm_conceptual_modeling.algo3.evaluation import parse_edge_list
from llm_conceptual_modeling.common.csv_schema import assert_required_columns
from llm_conceptual_modeling.common.types import PathLike


def write_failure_analysis(
    input_csv_path: PathLike,
    output_csv_path: PathLike,
    *,
    result_column: str,
) -> None:
    dataframe = pd.read_csv(input_csv_path)
    assert_required_columns(dataframe, [result_column], label="result columns")

    categories: list[str] = []
    parsed_edge_counts: list[int] = []
    is_failures: list[bool] = []

    for value in dataframe[result_column]:
        category, parsed_edge_count = _classify_result(value)
        categories.append(category)
        parsed_edge_counts.append(parsed_edge_count)
        is_failures.append(category != "valid_output")

    output = dataframe.copy()
    output["failure_category"] = categories
    output["parsed_edge_count"] = parsed_edge_counts
    output["is_failure"] = is_failures
    output.to_csv(output_csv_path, index=False)


def _classify_result(value: object) -> tuple[str, int]:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in {"nan", "none", "empty", "[]"}:
        return "empty_output", 0

    parsed_edges = parse_edge_list(text)
    if parsed_edges:
        return "valid_output", len(parsed_edges)

    return "malformed_output", 0
