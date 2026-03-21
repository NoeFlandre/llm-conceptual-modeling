import pandas as pd
from scipy.stats import ttest_rel

from llm_conceptual_modeling.cli import main


def test_cli_analyze_hypothesis_writes_paired_factor_tests(tmp_path) -> None:
    input_path = tmp_path / "evaluated.csv"
    output_path = tmp_path / "hypothesis.csv"
    dataframe = pd.DataFrame(
        {
            "Repetition": [0, 0, 1, 1, 2, 2, 3, 3],
            "Condition": ["x", "x", "x", "x", "x", "x", "x", "x"],
            "Explanation": [-1, 1, -1, 1, -1, 1, -1, 1],
            "accuracy": [0.1, 0.3, 0.4, 0.5, 0.2, 0.6, 0.3, 0.4],
        }
    )
    dataframe.to_csv(input_path, index=False)

    exit_code = main(
        [
            "analyze",
            "hypothesis",
            "--input",
            str(input_path),
            "--factor",
            "Explanation",
            "--pair-by",
            "Repetition",
            "--pair-by",
            "Condition",
            "--metric",
            "accuracy",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0

    actual = pd.read_csv(output_path)
    low_values = [0.1, 0.4, 0.2, 0.3]
    high_values = [0.3, 0.5, 0.6, 0.4]
    expected_statistic, expected_p_value = ttest_rel(high_values, low_values)

    expected = pd.DataFrame(
        {
            "source_input": [str(input_path)],
            "factor": ["Explanation"],
            "level_low": [-1],
            "level_high": [1],
            "metric": ["accuracy"],
            "pair_count": [4],
            "mean_low": [sum(low_values) / 4],
            "mean_high": [sum(high_values) / 4],
            "mean_difference": [(sum(high_values) - sum(low_values)) / 4],
            "t_statistic": [expected_statistic],
            "p_value": [expected_p_value],
            "p_value_adjusted": [expected_p_value],
            "correction_method": ["benjamini-hochberg"],
        }
    )

    pd.testing.assert_frame_equal(actual, expected)


def test_cli_analyze_hypothesis_rejects_missing_factor_column(tmp_path, capsys) -> None:
    input_path = tmp_path / "evaluated.csv"
    output_path = tmp_path / "hypothesis.csv"
    pd.DataFrame({"Repetition": [0, 0], "accuracy": [0.1, 0.2]}).to_csv(input_path, index=False)

    exit_code = main(
        [
            "analyze",
            "hypothesis",
            "--input",
            str(input_path),
            "--factor",
            "Explanation",
            "--pair-by",
            "Repetition",
            "--metric",
            "accuracy",
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Missing required factor columns" in captured.err
    assert not output_path.exists()


def test_cli_analyze_hypothesis_combines_inputs_and_adjusts_p_values(tmp_path) -> None:
    first_input_path = tmp_path / "evaluated1.csv"
    second_input_path = tmp_path / "evaluated2.csv"
    output_path = tmp_path / "hypothesis.csv"

    pd.DataFrame(
        {
            "Repetition": [0, 0, 1, 1, 2, 2, 3, 3],
            "Explanation": [-1, 1, -1, 1, -1, 1, -1, 1],
            "accuracy": [0.1, 0.8, 0.1, 0.7, 0.1, 0.9, 0.1, 0.8],
        }
    ).to_csv(first_input_path, index=False)
    pd.DataFrame(
        {
            "Repetition": [0, 0, 1, 1, 2, 2, 3, 3],
            "Explanation": [-1, 1, -1, 1, -1, 1, -1, 1],
            "accuracy": [0.2, 0.3, 0.2, 0.4, 0.2, 0.3, 0.2, 0.4],
        }
    ).to_csv(second_input_path, index=False)

    exit_code = main(
        [
            "analyze",
            "hypothesis",
            "--input",
            str(first_input_path),
            "--input",
            str(second_input_path),
            "--factor",
            "Explanation",
            "--pair-by",
            "Repetition",
            "--metric",
            "accuracy",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0

    actual = pd.read_csv(output_path)

    assert set(actual["source_input"]) == {str(first_input_path), str(second_input_path)}
    assert (actual["p_value_adjusted"] >= actual["p_value"]).all()
    assert (actual["correction_method"] == "benjamini-hochberg").all()
