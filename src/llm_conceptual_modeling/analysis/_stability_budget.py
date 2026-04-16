from __future__ import annotations

import pandas as pd


def build_replication_budget_overview_records(
    *,
    algorithm: str,
    budget_frame: pd.DataFrame,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    grouped_budget_frame = budget_frame.groupby("metric", dropna=False)
    for _, row in (
        grouped_budget_frame.agg(
            condition_count=("metric", "size"),
            max_required_total_runs=("required_total_runs", "max"),
            max_additional_runs_needed=("additional_runs_needed", "max"),
            mean_required_total_runs=("required_total_runs", "mean"),
            conditions_needing_more_runs=(
                "additional_runs_needed",
                lambda series: int((series > 0).sum()),
            ),
        )
        .reset_index()
        .iterrows()
    ):
        records.append(
            {
                "algorithm": algorithm,
                "metric": row["metric"],
                "condition_count": int(row["condition_count"]),
                "max_required_total_runs": int(row["max_required_total_runs"]),
                "max_additional_runs_needed": int(row["max_additional_runs_needed"]),
                "mean_required_total_runs": float(row["mean_required_total_runs"]),
                "conditions_needing_more_runs": int(row["conditions_needing_more_runs"]),
            }
        )
    return records
