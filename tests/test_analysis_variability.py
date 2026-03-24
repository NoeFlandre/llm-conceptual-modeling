import pandas as pd

from llm_conceptual_modeling.analysis.variability import write_output_variability_analysis


def test_write_output_variability_analysis_quantifies_edge_set_drift(tmp_path) -> None:
    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "variability.csv"
    pd.DataFrame(
        {
            "Condition": ["A", "A", "A", "B"],
            "Result": [
                "[('a', 'b')]",
                "[('a', 'b')]",
                "[('a', 'b'), ('b', 'c')]",
                "[('x', 'y')]",
            ],
        }
    ).to_csv(input_path, index=False)

    write_output_variability_analysis(
        [input_path],
        output_path,
        group_by=["Condition"],
        result_column="Result",
    )

    actual = pd.read_csv(output_path)

    group_a = actual.loc[actual["Condition"] == "A"].iloc[0]
    assert group_a["source_input"] == str(input_path)
    assert group_a["n_outputs"] == 3
    assert group_a["unique_output_count"] == 2
    assert group_a["mean_edge_count"] == 4 / 3
    assert round(group_a["sample_std_edge_count"], 6) == round(0.5773502691896257, 6)
    assert round(group_a["mean_pairwise_jaccard"], 6) == round(2 / 3, 6)
    assert round(group_a["min_pairwise_jaccard"], 6) == 0.5
    assert round(group_a["exact_match_pair_rate"], 6) == round(1 / 3, 6)
    assert group_a["union_edge_count"] == 2

    group_b = actual.loc[actual["Condition"] == "B"].iloc[0]
    assert group_b["n_outputs"] == 1
    assert group_b["unique_output_count"] == 1
    assert group_b["mean_pairwise_jaccard"] == 1.0
    assert group_b["exact_match_pair_rate"] == 1.0


def test_write_output_variability_analysis_rejects_missing_result_column(tmp_path) -> None:
    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "variability.csv"
    pd.DataFrame({"Condition": ["A"], "Other": ["[]"]}).to_csv(input_path, index=False)

    try:
        write_output_variability_analysis(
            [input_path],
            output_path,
            group_by=["Condition"],
            result_column="Result",
        )
    except ValueError as error:
        assert "Missing required result columns" in str(error)
    else:
        raise AssertionError("Expected ValueError for missing result column.")
