import pandas as pd

from llm_conceptual_modeling.common.csv_schema import assert_required_columns
from llm_conceptual_modeling.common.types import PathLike


def write_grouped_metric_summary(
    input_csv_paths: list[PathLike] | tuple[PathLike, ...],
    output_csv_path: PathLike,
    *,
    group_by: list[str],
    metrics: list[str],
) -> None:
    metric_frames: list[pd.DataFrame] = []
    for input_csv_path in input_csv_paths:
        dataframe = pd.read_csv(input_csv_path)
        assert_required_columns(dataframe, group_by, label="group-by columns")
        assert_required_columns(dataframe, metrics, label="metric columns")

        source_input = str(input_csv_path)
        for metric in metrics:
            summary = (
                dataframe.groupby(group_by, dropna=False)[metric]
                .agg(["count", "mean", "std", "median", "min", "max"])
                .reset_index()
                .rename(columns={"count": "n", "std": "sample_std"})
            )
            standard_error = summary["sample_std"] / summary["n"].pow(0.5)
            margin = 1.96 * standard_error.fillna(0.0)
            summary["ci95_low"] = summary["mean"] - margin
            summary["ci95_high"] = summary["mean"] + margin
            summary["metric"] = metric
            summary["source_input"] = source_input
            metric_frames.append(summary)

    output = pd.concat(metric_frames, ignore_index=True)
    ordered_columns = [
        "source_input",
        *group_by,
        "metric",
        "n",
        "mean",
        "sample_std",
        "median",
        "min",
        "max",
        "ci95_low",
        "ci95_high",
    ]
    output = output[ordered_columns]
    output.to_csv(output_csv_path, index=False)
