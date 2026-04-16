from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis._summary_helpers import (
    _build_metric_overview,
    _stringify_level,
)


def test_summary_helpers_build_metric_overview_and_stringify_level(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.csv"
    pd.DataFrame(
        {
            "source_input": ["a.csv", "a.csv"],
            "metric": ["accuracy", "accuracy"],
            "model": ["gpt-5", "gpt-5"],
            "Explanation": [-1, 1],
            "mean": [0.5, 0.9],
            "n": [2, 2],
        }
    ).to_csv(summary_path, index=False)

    overview = _build_metric_overview(
        summary_path=summary_path,
        algorithm="algo1",
        factor="Explanation",
    )

    assert overview.iloc[0]["level_low"] == "-1"
    assert overview.iloc[0]["level_high"] == "1"
    assert overview.iloc[0]["winner_count_low"] == 0
    assert overview.iloc[0]["winner_count_high"] == 1
    assert _stringify_level(2.0) == "2"
    assert _stringify_level("low") == "low"
