from __future__ import annotations

import pandas as pd


def _build_significance_summary(paired_tests: pd.DataFrame) -> pd.DataFrame:
    summary = paired_tests.copy()
    summary["is_significant"] = summary["p_value_adjusted"] <= 0.05
    summary["direction"] = summary.apply(_direction_label, axis=1)
    grouped_counts = summary.groupby(
        ["metric", "direction", "is_significant"],
        dropna=False,
    ).size()
    return grouped_counts.to_frame("test_count").reset_index()


def _build_factor_overview(
    *,
    paired_tests: pd.DataFrame,
    algorithm: str,
    factor: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for metric, metric_frame in paired_tests.groupby("metric", dropna=False):
        significant = metric_frame[metric_frame["p_value_adjusted"] <= 0.05]
        if significant.empty:
            strongest_row = metric_frame.sort_values(
                by=["p_value_adjusted", "p_value", "source_input"],
                na_position="last",
            ).iloc[0]
            strongest_source = str(strongest_row["source_input"])
            strongest_adjusted_p = strongest_row["p_value_adjusted"]
            strongest_effect_size = strongest_row["effect_size_paired_d"]
        else:
            strongest_row = significant.reindex(
                significant["effect_size_paired_d"].abs().sort_values(ascending=False).index
            ).iloc[0]
            strongest_source = str(strongest_row["source_input"])
            strongest_adjusted_p = strongest_row["p_value_adjusted"]
            strongest_effect_size = strongest_row["effect_size_paired_d"]

        records.append(
            {
                "algorithm": algorithm,
                "factor": factor,
                "metric": metric,
                "test_count": int(len(metric_frame)),
                "significant_test_count": int((metric_frame["p_value_adjusted"] <= 0.05).sum()),
                "significant_share": float((metric_frame["p_value_adjusted"] <= 0.05).mean()),
                "mean_difference_average": float(metric_frame["mean_difference"].mean()),
                "mean_absolute_effect_size": float(
                    metric_frame["effect_size_paired_d"].abs().mean()
                ),
                "positive_direction_count": int((metric_frame["mean_difference"] > 0).sum()),
                "negative_direction_count": int((metric_frame["mean_difference"] < 0).sum()),
                "null_direction_count": int((metric_frame["mean_difference"] == 0).sum()),
                "strongest_source_input": strongest_source,
                "strongest_adjusted_p_value": strongest_adjusted_p,
                "strongest_effect_size_paired_d": strongest_effect_size,
            }
        )
    return pd.DataFrame.from_records(records)


def _direction_label(row: pd.Series) -> str:
    mean_difference = row["mean_difference"]
    if pd.isna(mean_difference) or mean_difference == 0:
        return "equal"
    return "high_gt_low" if mean_difference > 0 else "high_lt_low"
