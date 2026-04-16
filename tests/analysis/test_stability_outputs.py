from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis._algo3_stability import Algo3PairAwareOutputs
from llm_conceptual_modeling.analysis._stability_outputs import (
    write_stability_bundle_outputs,
)


def test_write_stability_bundle_outputs_writes_metadata_and_readme(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()

    write_stability_bundle_outputs(
        output_dir_path=output_dir,
        manifest_records=[
            {
                "algorithm": "algo1",
                "factor": "condition",
                "relative_path": "algo1/condition_stability.csv",
                "description": "Condition stability",
            }
        ],
        overview_records=[
            {
                "algorithm": "algo1",
                "metric": "accuracy",
                "mean_cv": 0.01,
            }
        ],
        replication_budget_overview_records=[
            {
                "algorithm": "algo1",
                "metric": "accuracy",
                "condition_count": 2,
                "max_required_total_runs": 23,
                "max_additional_runs_needed": 18,
                "mean_required_total_runs": 14.0,
                "conditions_needing_more_runs": 1,
            }
        ],
        source_variability_frame=pd.DataFrame(
            [
                {
                    "algorithm": "algo1",
                    "metric": "accuracy",
                    "varying_condition_share": 0.5,
                }
            ]
        ),
        source_overall_frame=pd.DataFrame(
            [
                {
                    "algorithm": "algo1",
                    "metric": "accuracy",
                    "condition_count": 2,
                    "mean_cv": 0.01,
                    "median_cv": 0.01,
                    "max_cv": 0.02,
                    "mean_range_width": 0.03,
                    "max_range_width": 0.04,
                }
            ]
        ),
        algo3_pair_aware_outputs=Algo3PairAwareOutputs(
            condition_frame=pd.DataFrame(),
            depth_stability_frame=pd.DataFrame(),
            depth_variability_frame=pd.DataFrame(),
            number_of_words_stability_frame=pd.DataFrame(),
            number_of_words_variability_frame=pd.DataFrame(),
            overall_row={
                "algorithm": "algo3",
                "metric": "recall",
                "condition_count": 1,
                "mean_cv": 0.2,
                "median_cv": 0.2,
                "max_cv": 0.3,
                "mean_range_width": 0.4,
                "max_range_width": 0.5,
            },
            variability_row={
                "algorithm": "algo3",
                "metric": "recall",
                "varying_condition_share": 0.1,
            },
        ),
    )

    assert (output_dir / "bundle_manifest.csv").exists()
    assert (output_dir / "bundle_overview.csv").exists()
    assert (output_dir / "replication_budget_overview.csv").exists()
    assert (output_dir / "README.md").read_text(encoding="utf-8").startswith(
        "# Replication Stability Audit Bundle"
    )
