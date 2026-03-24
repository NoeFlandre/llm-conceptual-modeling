import pandas as pd

from llm_conceptual_modeling.cli import main


def test_cli_analyze_variability_writes_output_drift_metrics(tmp_path) -> None:
    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "variability.csv"
    pd.DataFrame(
        {
            "Condition": ["A", "A", "A"],
            "Result": [
                "[('a', 'b')]",
                "[('a', 'b')]",
                "[('a', 'b'), ('b', 'c')]",
            ],
        }
    ).to_csv(input_path, index=False)

    exit_code = main(
        [
            "analyze",
            "variability",
            "--input",
            str(input_path),
            "--group-by",
            "Condition",
            "--result-column",
            "Result",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0

    actual = pd.read_csv(output_path)
    assert list(actual.columns) == [
        "source_input",
        "Condition",
        "n_outputs",
        "unique_output_count",
        "mean_edge_count",
        "sample_std_edge_count",
        "mean_pairwise_jaccard",
        "min_pairwise_jaccard",
        "exact_match_pair_rate",
        "union_edge_count",
    ]
    assert actual.loc[0, "n_outputs"] == 3
