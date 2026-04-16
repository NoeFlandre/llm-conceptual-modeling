import pandas as pd

from llm_conceptual_modeling.analysis._hypothesis_bundle_helpers import (
    _build_factor_overview,
    _build_significance_summary,
)


def test_hypothesis_bundle_helpers_build_significance_and_overview() -> None:
    paired_tests = pd.DataFrame(
        {
            "metric": ["accuracy", "accuracy", "recall"],
            "p_value_adjusted": [0.01, 0.2, 0.04],
            "p_value": [0.01, 0.2, 0.04],
            "mean_difference": [0.5, -0.25, 0.1],
            "effect_size_paired_d": [1.2, -0.5, 0.8],
            "source_input": ["a.csv", "b.csv", "c.csv"],
        }
    )

    significance_summary = _build_significance_summary(paired_tests)
    factor_overview = _build_factor_overview(
        paired_tests=paired_tests,
        algorithm="algo1",
        factor="Explanation",
    )

    assert set(significance_summary["direction"]) == {"high_gt_low", "high_lt_low"}
    assert set(factor_overview["metric"]) == {"accuracy", "recall"}
    assert (
        factor_overview.loc[
            factor_overview["metric"] == "accuracy",
            "strongest_source_input",
        ].iloc[0]
        == "a.csv"
    )
