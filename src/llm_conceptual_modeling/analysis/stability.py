import pandas as pd

from llm_conceptual_modeling.common.csv_schema import assert_required_columns
from llm_conceptual_modeling.common.types import PathLike


def write_grouped_metric_stability(
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
            stability = (
                dataframe.groupby(group_by, dropna=False)[metric]
                .agg(["count", "mean", "std", "min", "max"])
                .reset_index()
                .rename(columns={"count": "n", "std": "sample_std"})
            )
            stability["range_width"] = stability["max"] - stability["min"]
            stability["coefficient_of_variation"] = stability["sample_std"] / stability["mean"]
            stability["metric"] = metric
            stability["source_input"] = source_input
            metric_frames.append(stability)

    output = pd.concat(metric_frames, ignore_index=True)
    ordered_columns = [
        "source_input",
        *group_by,
        "metric",
        "n",
        "mean",
        "sample_std",
        "min",
        "max",
        "range_width",
        "coefficient_of_variation",
    ]
    output = output[ordered_columns]
    output.to_csv(output_csv_path, index=False)
