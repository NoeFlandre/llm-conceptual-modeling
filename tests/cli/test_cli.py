import pandas as pd

from llm_conceptual_modeling.cli import main


def test_cli_analyze_summary_bundle_writes_organized_review_artifacts(tmp_path) -> None:
    results_root = tmp_path / "results"
    _copy_fixture(
        "tests/reference_fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg1_sg2.csv",
        results_root / "algo1" / "gpt-5" / "evaluated" / "metrics_sg1_sg2.csv",
    )
    _copy_fixture(
        "tests/reference_fixtures/legacy/algo2/gpt-5/evaluated/metrics_sg1_sg2.csv",
        results_root / "algo2" / "gpt-5" / "evaluated" / "metrics_sg1_sg2.csv",
    )
    _copy_fixture(
        "tests/reference_fixtures/legacy/algo3/gpt-5/evaluated/method3_results_evaluated_gpt5.csv",
        results_root / "algo3" / "gpt-5" / "evaluated" / "method3_results_evaluated_gpt5.csv",
    )
    output_dir = tmp_path / "bundle"

    exit_code = main(
        [
            "analyze",
            "summary-bundle",
            "--results-root",
            str(results_root),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "bundle_manifest.csv").exists()
    assert (output_dir / "bundle_overview.csv").exists()
    assert (output_dir / "algo1" / "explanation" / "grouped_metric_summary.csv").exists()
    assert (output_dir / "algo2" / "convergence" / "metric_overview.csv").exists()
    assert (output_dir / "algo3" / "depth" / "metric_overview.csv").exists()


def test_cli_eval_algo1_writes_legacy_parity_metrics(tmp_path) -> None:
    raw_path = "tests/reference_fixtures/legacy/algo1/gpt-5/raw/algorithm1_results_sg1_sg2.csv"
    expected_path = "tests/reference_fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg1_sg2.csv"
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


def test_cli_baseline_algo1_writes_raw_results_for_requested_pair(tmp_path) -> None:
    output_path = tmp_path / "algorithm1_baseline_sg1_sg2.csv"

    exit_code = main(
        [
            "baseline",
            "algo1",
            "--pair",
            "sg1_sg2",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0

    actual = pd.read_csv(output_path)

    assert len(actual) == 160
    assert actual["Repetition"].tolist()[:5] == [0, 0, 0, 0, 0]
    assert sorted(actual["Explanation"].unique().tolist()) == [-1, 1]
    assert sorted(actual["Example"].unique().tolist()) == [-1, 1]
    assert sorted(actual["Counterexample"].unique().tolist()) == [-1, 1]
    assert sorted(actual["Array/List(1/-1)"].unique().tolist()) == [-1, 1]
    assert sorted(actual["Tag/Adjacency(1/-1)"].unique().tolist()) == [-1, 1]
    assert actual["Result"].nunique() == 1


def test_cli_baseline_algo1_output_is_evaluable(tmp_path) -> None:
    raw_output_path = tmp_path / "algorithm1_baseline_sg1_sg2.csv"
    evaluated_output_path = tmp_path / "metrics.csv"

    baseline_exit_code = main(
        [
            "baseline",
            "algo1",
            "--pair",
            "sg1_sg2",
            "--output",
            str(raw_output_path),
        ]
    )
    eval_exit_code = main(
        [
            "eval",
            "algo1",
            "--input",
            str(raw_output_path),
            "--output",
            str(evaluated_output_path),
        ]
    )

    assert baseline_exit_code == 0
    assert eval_exit_code == 0

    actual = pd.read_csv(evaluated_output_path)

    assert len(actual) == 160
    assert {"accuracy", "recall", "precision", "f1"}.issubset(actual.columns)


def test_cli_baseline_algo2_output_is_evaluable(tmp_path) -> None:
    raw_output_path = tmp_path / "algorithm2_baseline_sg1_sg2.csv"
    evaluated_output_path = tmp_path / "metrics.csv"

    baseline_exit_code = main(
        [
            "baseline",
            "algo2",
            "--pair",
            "sg1_sg2",
            "--output",
            str(raw_output_path),
        ]
    )
    eval_exit_code = main(
        [
            "eval",
            "algo2",
            "--input",
            str(raw_output_path),
            "--output",
            str(evaluated_output_path),
        ]
    )

    assert baseline_exit_code == 0
    assert eval_exit_code == 0

    actual = pd.read_csv(evaluated_output_path)

    assert len(actual) == 320
    assert {"accuracy", "recall", "precision", "f1"}.issubset(actual.columns)
    assert sorted(actual["Convergence"].unique().tolist()) == [-1, 1]


def test_cli_baseline_algo3_output_is_evaluable(tmp_path) -> None:
    raw_output_path = tmp_path / "method3_baseline.csv"
    evaluated_output_path = tmp_path / "method3_results_evaluated.csv"

    baseline_exit_code = main(
        [
            "baseline",
            "algo3",
            "--pair",
            "subgraph_1_to_subgraph_3",
            "--output",
            str(raw_output_path),
        ]
    )
    eval_exit_code = main(
        [
            "eval",
            "algo3",
            "--input",
            str(raw_output_path),
            "--output",
            str(evaluated_output_path),
        ]
    )

    assert baseline_exit_code == 0
    assert eval_exit_code == 0

    actual = pd.read_csv(evaluated_output_path)

    assert len(actual) == 80
    assert "Recall" in actual.columns
    assert sorted(actual["Depth"].unique().tolist()) == [1, 2]
    assert sorted(actual["Number of Words"].unique().tolist()) == [3, 5]


def test_cli_eval_algo2_writes_legacy_parity_metrics(tmp_path) -> None:
    raw_path = "tests/reference_fixtures/legacy/algo2/gpt-5/raw/algorithm2_results_sg1_sg2.csv"
    expected_path = "tests/reference_fixtures/legacy/algo2/gpt-5/evaluated/metrics_sg1_sg2.csv"
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
    raw_path = "tests/reference_fixtures/legacy/algo3/gpt-5/raw/method3_results_gpt5.csv"
    expected_path = (
        "tests/reference_fixtures/legacy/algo3/gpt-5/evaluated/method3_results_evaluated_gpt5.csv"
    )
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
        "tests/reference_fixtures/legacy/algo1/gpt-5/factorial/"
        "factorial_analysis_algo1_gpt_5_without_error.csv"
    )
    output_path = tmp_path / "factorial.csv"

    exit_code = main(
        [
            "factorial",
            "algo1",
            "--input",
            "tests/reference_fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg1_sg2.csv",
            "--input",
            "tests/reference_fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg2_sg3.csv",
            "--input",
            "tests/reference_fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg3_sg1.csv",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0

    actual = pd.read_csv(output_path)
    expected = pd.read_csv(expected_path)
    pd.testing.assert_frame_equal(actual, expected)


def _copy_fixture(source: str, destination) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.read_csv(source).to_csv(destination, index=False)


def test_cli_factorial_algo2_writes_legacy_parity_output(tmp_path) -> None:
    expected_path = (
        "tests/reference_fixtures/legacy/algo2/gpt-5/factorial/"
        "factorial_analysis_gpt_5_algo2_without_error.csv"
    )
    output_path = tmp_path / "factorial.csv"

    exit_code = main(
        [
            "factorial",
            "algo2",
            "--input",
            "tests/reference_fixtures/legacy/algo2/gpt-5/evaluated/metrics_sg1_sg2.csv",
            "--input",
            "tests/reference_fixtures/legacy/algo2/gpt-5/evaluated/metrics_sg2_sg3.csv",
            "--input",
            "tests/reference_fixtures/legacy/algo2/gpt-5/evaluated/metrics_sg3_sg1.csv",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0

    actual = pd.read_csv(output_path)
    expected = pd.read_csv(expected_path)
    pd.testing.assert_frame_equal(actual, expected)


def test_cli_factorial_algo3_writes_legacy_parity_output(tmp_path) -> None:
    expected_path = (
        "tests/reference_fixtures/legacy/algo3/gpt-5/factorial/"
        "factorial_analysis_results_gpt5_without_error.csv"
    )
    output_path = tmp_path / "factorial.csv"

    exit_code = main(
        [
            "factorial",
            "algo3",
            "--input",
            "tests/reference_fixtures/legacy/algo3/gpt-5/evaluated/method3_results_evaluated_gpt5.csv",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0

    actual = pd.read_csv(output_path)
    expected = pd.read_csv(expected_path)
    pd.testing.assert_frame_equal(actual, expected)
