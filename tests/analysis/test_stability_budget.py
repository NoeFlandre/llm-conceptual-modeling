from __future__ import annotations

import pandas as pd

from llm_conceptual_modeling.analysis._stability_budget import (
    build_replication_budget_overview_records,
)


def test_build_replication_budget_overview_records_groups_by_metric() -> None:
    budget_frame = pd.DataFrame(
        [
            {
                "metric": "accuracy",
                "required_total_runs": 23,
                "additional_runs_needed": 18,
            },
            {
                "metric": "accuracy",
                "required_total_runs": 5,
                "additional_runs_needed": 0,
            },
            {
                "metric": "recall",
                "required_total_runs": 11,
                "additional_runs_needed": 6,
            },
        ]
    )

    records = build_replication_budget_overview_records(
        algorithm="algo1",
        budget_frame=budget_frame,
    )

    assert records == [
        {
            "algorithm": "algo1",
            "metric": "accuracy",
            "condition_count": 2,
            "max_required_total_runs": 23,
            "max_additional_runs_needed": 18,
            "mean_required_total_runs": 14.0,
            "conditions_needing_more_runs": 1,
        },
        {
            "algorithm": "algo1",
            "metric": "recall",
            "condition_count": 1,
            "max_required_total_runs": 11,
            "max_additional_runs_needed": 6,
            "mean_required_total_runs": 11.0,
            "conditions_needing_more_runs": 1,
        },
    ]
