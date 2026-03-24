import pandas as pd

from llm_conceptual_modeling.cli import main


def test_cli_analyze_stability_writes_grouped_variability_statistics(tmp_path) -> None:
    input_path = tmp_path / "evaluated.csv"
    output_path = tmp_path / "stability.csv"
    pd.DataFrame(
        {
            "Condition": ["A", "A", "A", "B", "B"],
            "accuracy": [0.2, 0.4, 0.6, 1.0, 1.0],
        }
    ).to_csv(input_path, index=False)

    exit_code = main(
        [
            "analyze",
            "stability",
            "--input",
            str(input_path),
            "--group-by",
            "Condition",
            "--metric",
            "accuracy",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0

    actual = pd.read_csv(output_path)

    expected = pd.read_csv(input_path)
    expected = (
        expected.groupby("Condition", dropna=False)["accuracy"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"count": "n", "std": "sample_std"})
    )
    expected["range_width"] = expected["max"] - expected["min"]
    expected["coefficient_of_variation"] = expected["sample_std"] / expected["mean"]
    expected["metric"] = "accuracy"
    expected["source_input"] = str(input_path)

    expected = expected[
        [
            "source_input",
            "Condition",
            "metric",
            "n",
            "mean",
            "sample_std",
            "min",
            "max",
            "range_width",
            "coefficient_of_variation",
        ]
    ]

    pd.testing.assert_frame_equal(actual, expected)


def test_cli_analyze_stability_rejects_missing_metric_column(tmp_path, capsys) -> None:
    input_path = tmp_path / "evaluated.csv"
    output_path = tmp_path / "stability.csv"
    pd.DataFrame({"Condition": ["A"], "accuracy": [0.5]}).to_csv(input_path, index=False)

    exit_code = main(
        [
            "analyze",
            "stability",
            "--input",
            str(input_path),
            "--group-by",
            "Condition",
            "--metric",
            "precision",
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Missing required metric columns" in captured.err
    assert not output_path.exists()


def test_cli_analyze_stability_combines_multiple_inputs_with_source_labels(tmp_path) -> None:
    first_input_path = tmp_path / "evaluated1.csv"
    second_input_path = tmp_path / "evaluated2.csv"
    output_path = tmp_path / "stability.csv"
    pd.DataFrame({"Condition": ["A", "A"], "accuracy": [0.2, 0.4]}).to_csv(
        first_input_path, index=False
    )
    pd.DataFrame({"Condition": ["A", "A"], "accuracy": [0.1, 0.1]}).to_csv(
        second_input_path, index=False
    )

    exit_code = main(
        [
            "analyze",
            "stability",
            "--input",
            str(first_input_path),
            "--input",
            str(second_input_path),
            "--group-by",
            "Condition",
            "--metric",
            "accuracy",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0

    actual = pd.read_csv(output_path)

    assert "source_input" in actual.columns
    assert set(actual["source_input"]) == {str(first_input_path), str(second_input_path)}
    assert len(actual) == 2
