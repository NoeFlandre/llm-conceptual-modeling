from collections.abc import Sequence

import pandas as pd
from scipy.stats import ttest_rel

from llm_conceptual_modeling.common.csv_schema import assert_required_columns
from llm_conceptual_modeling.common.types import PathLike


def write_paired_factor_hypothesis_tests(
    input_csv_paths: list[PathLike] | tuple[PathLike, ...],
    output_csv_path: PathLike,
    *,
    factor: str,
    pair_by: list[str],
    metrics: list[str],
) -> None:
    frames: list[pd.DataFrame] = []
    for input_csv_path in input_csv_paths:
        dataframe = pd.read_csv(input_csv_path)
        assert_required_columns(dataframe, [factor], label="factor columns")
        assert_required_columns(dataframe, pair_by, label="pair-by columns")
        assert_required_columns(dataframe, metrics, label="metric columns")

        levels = sorted(dataframe[factor].dropna().unique().tolist())
        if len(levels) != 2:
            raise ValueError(
                f"Factor column '{factor}' must have exactly two non-null levels, found {levels}"
            )

        source_input = str(input_csv_path)
        for metric in metrics:
            pair_frame = _build_pair_frame(dataframe, factor=factor, pair_by=pair_by, metric=metric)
            low_values = pair_frame["value_low"]
            high_values = pair_frame["value_high"]
            statistic, p_value = ttest_rel(high_values, low_values)
            frames.append(
                pd.DataFrame(
                    {
                        "source_input": [source_input],
                        "factor": [factor],
                        "level_low": [levels[0]],
                        "level_high": [levels[1]],
                        "metric": [metric],
                        "pair_count": [len(pair_frame)],
                        "mean_low": [low_values.mean()],
                        "mean_high": [high_values.mean()],
                        "mean_difference": [(high_values - low_values).mean()],
                        "t_statistic": [statistic],
                        "p_value": [p_value],
                    }
                )
            )

    output = pd.concat(frames, ignore_index=True)
    output["p_value_adjusted"] = _benjamini_hochberg(output["p_value"].tolist())
    output["correction_method"] = "benjamini-hochberg"
    output.to_csv(output_csv_path, index=False)


def _build_pair_frame(
    dataframe: pd.DataFrame,
    *,
    factor: str,
    pair_by: list[str],
    metric: str,
) -> pd.DataFrame:
    levels = sorted(dataframe[factor].dropna().unique().tolist())
    indexed = dataframe[pair_by + [factor, metric]].copy()
    pair_frame = indexed.pivot(
        index=pair_by,
        columns=factor,
        values=metric,
    ).reset_index()
    if pair_frame.shape[0] == 0:
        raise ValueError("No paired observations available for hypothesis test")

    expected_columns = set(levels)
    observed_columns = set(pair_frame.columns) - set(pair_by)
    if observed_columns != expected_columns:
        raise ValueError(
            f"Paired observations for factor '{factor}' are incomplete; expected levels {levels}"
        )

    pair_frame = pair_frame.rename(columns={levels[0]: "value_low", levels[1]: "value_high"})
    if pair_frame[["value_low", "value_high"]].isna().any().any():
        raise ValueError(
            f"Paired observations for factor '{factor}' are incomplete for metric '{metric}'"
        )
    return pair_frame


def _benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    count = len(p_values)
    ranked = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * count
    running_min = 1.0
    for rank, (index, p_value) in enumerate(reversed(ranked), start=1):
        denominator = count - rank + 1
        candidate = min(1.0, p_value * count / denominator)
        running_min = min(running_min, candidate)
        adjusted[index] = running_min
    return adjusted
