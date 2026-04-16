from __future__ import annotations

from pathlib import Path

import pandas as pd


def _build_metric_overview(*, summary_path: Path, algorithm: str, factor: str) -> pd.DataFrame:
    summary = pd.read_csv(summary_path)
    level_column = factor
    records: list[dict[str, object]] = []

    for metric, metric_frame in summary.groupby("metric", dropna=False):
        levels = sorted(metric_frame[level_column].drop_duplicates().tolist())
        if len(levels) != 2:
            raise ValueError(
                f"Expected exactly two levels for {algorithm} {factor} {metric}, found {levels}"
            )
        low_level, high_level = levels

        pivot = metric_frame.pivot(index="source_input", columns=level_column, values="mean")
        low_wins = 0
        high_wins = 0
        ties = 0
        for _, row in pivot.iterrows():
            low_value = float(row[low_level])
            high_value = float(row[high_level])
            if high_value > low_value:
                high_wins += 1
            elif high_value < low_value:
                low_wins += 1
            else:
                ties += 1

        weighted_means = (
            metric_frame.assign(weighted_sum=metric_frame["mean"] * metric_frame["n"])
            .groupby(level_column, dropna=False)[["weighted_sum", "n"]]
            .sum()
        )
        low_global_mean = float(
            weighted_means.loc[low_level, "weighted_sum"] / weighted_means.loc[low_level, "n"]
        )
        high_global_mean = float(
            weighted_means.loc[high_level, "weighted_sum"] / weighted_means.loc[high_level, "n"]
        )

        records.append(
            {
                "algorithm": algorithm,
                "factor": factor,
                "metric": metric,
                "source_file_count": int(metric_frame["source_input"].nunique()),
                "level_low": _stringify_level(low_level),
                "level_high": _stringify_level(high_level),
                "global_mean_low": low_global_mean,
                "global_mean_high": high_global_mean,
                "global_mean_difference_high_minus_low": high_global_mean - low_global_mean,
                "winner_count_low": low_wins,
                "winner_count_high": high_wins,
                "tie_count": ties,
            }
        )

    return pd.DataFrame.from_records(records)


def _stringify_level(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
