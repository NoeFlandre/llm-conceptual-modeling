import pandas as pd

from llm_conceptual_modeling.analysis._figure_stats import (
    _compute_distributional_summary,
    _melt_to_long,
)


def test_figure_stats_build_long_rows_and_distributional_summary() -> None:
    frame = pd.DataFrame(
        {
            "Repetition": [0, 1],
            "Explanation": [1, -1],
            "Example": [1, -1],
            "accuracy": [0.9, 0.5],
            "precision": [0.4, 0.2],
            "recall": [0.1, 0.05],
        }
    )

    long_frame = _melt_to_long(
        frame,
        model="gpt-5",
        source_input="/tmp/results/algo1/gpt-5/evaluated/x.csv",
        algorithm="algo1",
        id_columns=["Repetition", "Explanation", "Example"],
        metrics=["accuracy", "precision", "recall"],
    )
    summary = _compute_distributional_summary(
        long_frame,
        algorithm="algo1",
        model="gpt-5",
    )

    assert set(long_frame["metric"]) == {"accuracy", "precision", "recall"}
    assert set(summary["metric"]) == {"accuracy", "precision", "recall"}
    assert summary.loc[summary["metric"] == "accuracy", "n"].iloc[0] == 2
