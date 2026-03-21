from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.common.csv_schema import assert_required_columns
from llm_conceptual_modeling.common.types import PathLike


def write_figure_ready_metric_rows(
    input_csv_paths: list[PathLike] | tuple[PathLike, ...],
    output_csv_path: PathLike,
    *,
    id_columns: list[str],
    metrics: list[str],
) -> None:
    frames: list[pd.DataFrame] = []
    for input_csv_path in input_csv_paths:
        dataframe = pd.read_csv(input_csv_path)
        assert_required_columns(dataframe, id_columns, label="id columns")
        assert_required_columns(dataframe, metrics, label="metric columns")

        algorithm, model = _infer_result_metadata(input_csv_path)
        figure_rows = dataframe[id_columns + metrics].melt(
            id_vars=id_columns,
            value_vars=metrics,
            var_name="metric",
            value_name="value",
        )
        figure_rows.insert(0, "model", model)
        figure_rows.insert(0, "algorithm", algorithm)
        figure_rows.insert(0, "source_input", str(input_csv_path))
        frames.append(figure_rows)

    output = pd.concat(frames, ignore_index=True)
    output.to_csv(output_csv_path, index=False)


def _infer_result_metadata(input_csv_path: PathLike) -> tuple[str, str]:
    parts = Path(input_csv_path).parts
    for index, part in enumerate(parts):
        if part == "results" and index + 2 < len(parts):
            return parts[index + 1], parts[index + 2]
        if part == "legacy" and index + 2 < len(parts):
            return parts[index + 1], parts[index + 2]
    return "unknown", "unknown"
