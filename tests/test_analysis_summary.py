import pandas as pd

from llm_conceptual_modeling.cli import main


def test_cli_analyze_summary_writes_grouped_metric_statistics(tmp_path) -> None:
    input_path = "tests/fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg1_sg2.csv"
    output_path = tmp_path / "summary.csv"

    exit_code = main(
        [
            "analyze",
            "summary",
            "--input",
            input_path,
            "--group-by",
            "Explanation",
            "--metric",
            "accuracy",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0

    actual = pd.read_csv(output_path)
    source = pd.read_csv(input_path)
    expected = (
        source.groupby(["Explanation"], dropna=False)["accuracy"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
        .rename(columns={"count": "n", "std": "sample_std"})
    )
    standard_error = expected["sample_std"] / expected["n"].pow(0.5)
    margin = 1.96 * standard_error.fillna(0.0)
    expected["ci95_low"] = expected["mean"] - margin
    expected["ci95_high"] = expected["mean"] + margin
    expected["metric"] = "accuracy"

    expected = expected[
        [
            "Explanation",
            "metric",
            "n",
            "mean",
            "sample_std",
            "median",
            "min",
            "max",
            "ci95_low",
            "ci95_high",
        ]
    ].sort_values("Explanation", ignore_index=True)
    actual = actual.sort_values("Explanation", ignore_index=True)

    pd.testing.assert_frame_equal(actual, expected)


def test_cli_analyze_summary_rejects_missing_metric_column(tmp_path, capsys) -> None:
    output_path = tmp_path / "summary.csv"

    exit_code = main(
        [
            "analyze",
            "summary",
            "--input",
            "tests/fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg1_sg2.csv",
            "--group-by",
            "Explanation",
            "--metric",
            "not_a_metric",
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Missing required metric columns" in captured.err
    assert not output_path.exists()
