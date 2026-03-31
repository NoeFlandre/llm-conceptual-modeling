from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.common.factorial_core import run_generalized_factorial_analysis
from llm_conceptual_modeling.common.types import GeneralizedFactorialSpec


def test_generalized_factorial_analysis_includes_interactions_for_decoding_factors(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "evaluated.csv"
    output_path = tmp_path / "factorial.csv"
    pd.DataFrame(
        [
            {
                "Explanation": -1,
                "Decoding Algorithm": "greedy",
                "Beam Width Level": 0,
                "Contrastive Penalty Level": 0,
                "Repetition": 0,
                "accuracy": 0.1,
            },
            {
                "Explanation": 1,
                "Decoding Algorithm": "beam",
                "Beam Width Level": -1,
                "Contrastive Penalty Level": 0,
                "Repetition": 0,
                "accuracy": 0.9,
            },
            {
                "Explanation": -1,
                "Decoding Algorithm": "contrastive",
                "Beam Width Level": 0,
                "Contrastive Penalty Level": -1,
                "Repetition": 0,
                "accuracy": 0.3,
            },
            {
                "Explanation": 1,
                "Decoding Algorithm": "beam",
                "Beam Width Level": 1,
                "Contrastive Penalty Level": 0,
                "Repetition": 1,
                "accuracy": 0.7,
            },
            {
                "Explanation": -1,
                "Decoding Algorithm": "contrastive",
                "Beam Width Level": 0,
                "Contrastive Penalty Level": 1,
                "Repetition": 1,
                "accuracy": 0.4,
            },
            {
                "Explanation": 1,
                "Decoding Algorithm": "greedy",
                "Beam Width Level": 0,
                "Contrastive Penalty Level": 0,
                "Repetition": 1,
                "accuracy": 0.2,
            },
        ]
    ).to_csv(input_path, index=False)

    run_generalized_factorial_analysis(
        [input_path],
        output_path,
        GeneralizedFactorialSpec(
            factor_columns=[
                "Explanation",
                "Decoding Algorithm",
                "Beam Width Level",
                "Contrastive Penalty Level",
            ],
            metric_columns=["accuracy"],
            output_columns=["accuracy", "Feature"],
            replication_column="Repetition",
        ),
    )

    actual = pd.read_csv(output_path)
    interaction_features = [
        feature for feature in actual["Feature"].tolist() if "_AND_" in str(feature)
    ]

    assert any("Decoding Algorithm" in str(feature) for feature in interaction_features)
    assert any("Beam Width Level" in str(feature) for feature in interaction_features)
