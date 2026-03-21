import pandas as pd

from llm_conceptual_modeling.cli import main


def test_cli_eval_algo1_writes_legacy_parity_metrics(tmp_path) -> None:
    raw_path = "tests/fixtures/legacy/algo1/gpt-5/raw/algorithm1_results_sg1_sg2.csv"
    expected_path = "tests/fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg1_sg2.csv"
    output_path = tmp_path / "metrics.csv"

    exit_code = main(
        [
            "eval",
            "algo1",
            "--input",
            raw_path,
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0

    actual = pd.read_csv(output_path)
    expected = pd.read_csv(expected_path)
    pd.testing.assert_series_equal(actual["accuracy"], expected["accuracy"], check_names=False)


def test_cli_eval_algo2_writes_legacy_parity_metrics(tmp_path) -> None:
    raw_path = "tests/fixtures/legacy/algo2/gpt-5/raw/algorithm2_results_sg1_sg2.csv"
    expected_path = "tests/fixtures/legacy/algo2/gpt-5/evaluated/metrics_sg1_sg2.csv"
    output_path = tmp_path / "metrics.csv"

    exit_code = main(
        [
            "eval",
            "algo2",
            "--input",
            raw_path,
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0

    actual = pd.read_csv(output_path)
    expected = pd.read_csv(expected_path)
    pd.testing.assert_series_equal(actual["accuracy"], expected["accuracy"], check_names=False)
