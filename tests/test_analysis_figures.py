import pandas as pd

from llm_conceptual_modeling.cli import main


def test_cli_analyze_figures_writes_tidy_metric_rows(tmp_path) -> None:
    input_path = tmp_path / "data" / "results" / "algo1" / "model-a" / "evaluated" / "metrics.csv"
    input_path.parent.mkdir(parents=True)
    output_path = tmp_path / "figures.csv"
    pd.DataFrame(
        {
            "Repetition": [0, 1],
            "Explanation": [-1, 1],
            "accuracy": [0.2, 0.4],
            "recall": [0.5, 0.7],
        }
    ).to_csv(input_path, index=False)

    exit_code = main(
        [
            "analyze",
            "figures",
            "--input",
            str(input_path),
            "--id-column",
            "Repetition",
            "--id-column",
            "Explanation",
            "--metric",
            "accuracy",
            "--metric",
            "recall",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0

    actual = pd.read_csv(output_path)
    expected = pd.DataFrame(
        {
            "source_input": [str(input_path)] * 4,
            "algorithm": ["algo1"] * 4,
            "model": ["model-a"] * 4,
            "Repetition": [0, 1, 0, 1],
            "Explanation": [-1, 1, -1, 1],
            "metric": ["accuracy", "accuracy", "recall", "recall"],
            "value": [0.2, 0.4, 0.5, 0.7],
        }
    )

    pd.testing.assert_frame_equal(actual, expected)


def test_cli_analyze_figures_rejects_missing_metric_column(tmp_path, capsys) -> None:
    input_path = tmp_path / "evaluated.csv"
    output_path = tmp_path / "figures.csv"
    pd.DataFrame({"Repetition": [0], "accuracy": [0.5]}).to_csv(input_path, index=False)

    exit_code = main(
        [
            "analyze",
            "figures",
            "--input",
            str(input_path),
            "--id-column",
            "Repetition",
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


def test_cli_analyze_figures_combines_multiple_inputs_with_unknown_path_metadata(tmp_path) -> None:
    first_input_path = tmp_path / "evaluated1.csv"
    second_input_path = tmp_path / "evaluated2.csv"
    output_path = tmp_path / "figures.csv"
    pd.DataFrame({"Repetition": [0], "accuracy": [0.2]}).to_csv(first_input_path, index=False)
    pd.DataFrame({"Repetition": [1], "accuracy": [0.4]}).to_csv(second_input_path, index=False)

    exit_code = main(
        [
            "analyze",
            "figures",
            "--input",
            str(first_input_path),
            "--input",
            str(second_input_path),
            "--id-column",
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
    assert set(actual["algorithm"]) == {"unknown"}
    assert set(actual["model"]) == {"unknown"}
    assert len(actual) == 2
