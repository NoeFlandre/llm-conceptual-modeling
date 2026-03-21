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


def test_cli_eval_algo3_writes_legacy_parity_recall(tmp_path) -> None:
    raw_path = "tests/fixtures/legacy/algo3/gpt-5/raw/method3_results_gpt5.csv"
    expected_path = "tests/fixtures/legacy/algo3/gpt-5/evaluated/method3_results_evaluated_gpt5.csv"
    output_path = tmp_path / "method3_results_evaluated_gpt5.csv"

    exit_code = main(
        [
            "eval",
            "algo3",
            "--input",
            raw_path,
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0

    actual = pd.read_csv(output_path)
    expected = pd.read_csv(expected_path)
    pd.testing.assert_series_equal(actual["Recall"], expected["Recall"], check_names=False)


def test_cli_factorial_algo1_writes_legacy_parity_output(tmp_path) -> None:
    expected_path = (
        "tests/fixtures/legacy/algo1/gpt-5/factorial/factorial_analysis_algo1_gpt_5_without_error.csv"
    )
    output_path = tmp_path / "factorial.csv"

    exit_code = main(
        [
            "factorial",
            "algo1",
            "--input",
            "tests/fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg1_sg2.csv",
            "--input",
            "tests/fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg2_sg3.csv",
            "--input",
            "tests/fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg3_sg1.csv",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0

    actual = pd.read_csv(output_path)
    expected = pd.read_csv(expected_path)
    pd.testing.assert_frame_equal(actual, expected)
