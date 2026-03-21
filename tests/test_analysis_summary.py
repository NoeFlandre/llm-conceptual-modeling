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
    expected["source_input"] = input_path

    expected = expected[
        [
            "source_input",
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


def test_cli_analyze_summary_combines_multiple_inputs_with_source_labels(tmp_path) -> None:
    output_path = tmp_path / "summary.csv"

    exit_code = main(
        [
            "analyze",
            "summary",
            "--input",
            "tests/fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg1_sg2.csv",
            "--input",
            "tests/fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg2_sg3.csv",
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

    assert "source_input" in actual.columns
    assert set(actual["source_input"]) == {
        "tests/fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg1_sg2.csv",
        "tests/fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg2_sg3.csv",
    }
    assert len(actual) == 4


def test_cli_analyze_baseline_comparison_writes_model_deltas(tmp_path) -> None:
    baseline_dir = (
        tmp_path / "data" / "baselines" / "direct-cross-graph" / "algo1" / "evaluated"
    )
    llm_dir = tmp_path / "data" / "results" / "algo1" / "model-x" / "evaluated"
    baseline_pair_a_path = baseline_dir / "metrics_sg1_sg2.csv"
    baseline_pair_b_path = baseline_dir / "metrics_sg2_sg3.csv"
    llm_pair_a_path = llm_dir / "metrics_sg1_sg2.csv"
    llm_pair_b_path = llm_dir / "metrics_sg2_sg3.csv"
    output_path = tmp_path / "comparison.csv"

    baseline_dir.mkdir(parents=True, exist_ok=True)
    llm_dir.mkdir(parents=True, exist_ok=True)

    baseline_pair_a_frame = pd.DataFrame({"accuracy": [0.2, 0.4], "precision": [0.3, 0.5]})
    baseline_pair_b_frame = pd.DataFrame({"accuracy": [0.4, 0.8], "precision": [0.7, 0.9]})
    llm_pair_a_frame = pd.DataFrame({"accuracy": [0.6, 0.8], "precision": [0.8, 1.0]})
    llm_pair_b_frame = pd.DataFrame({"accuracy": [0.8, 1.0], "precision": [0.9, 1.0]})

    baseline_pair_a_frame.to_csv(baseline_pair_a_path, index=False)
    baseline_pair_b_frame.to_csv(baseline_pair_b_path, index=False)
    llm_pair_a_frame.to_csv(llm_pair_a_path, index=False)
    llm_pair_b_frame.to_csv(llm_pair_b_path, index=False)

    exit_code = main(
        [
            "analyze",
            "baseline-comparison",
            "--baseline-input",
            str(baseline_pair_a_path),
            "--baseline-input",
            str(baseline_pair_b_path),
            "--input",
            str(llm_pair_a_path),
            "--input",
            str(llm_pair_b_path),
            "--metric",
            "accuracy",
            "--metric",
            "precision",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0

    actual = pd.read_csv(output_path).sort_values("metric", ignore_index=True)

    assert actual["algorithm"].tolist() == ["algo1", "algo1"]
    assert actual["model"].tolist() == ["model-x", "model-x"]
    assert actual["matched_file_count"].tolist() == [2, 2]
    assert actual["metric"].tolist() == ["accuracy", "precision"]
    assert actual["baseline_mean"].round(6).tolist() == [0.45, 0.6]
    assert actual["llm_mean"].round(6).tolist() == [0.8, 0.925]
    assert actual["mean_delta"].round(6).tolist() == [0.35, 0.325]


def test_cli_analyze_baseline_comparison_reuses_single_baseline_input(tmp_path) -> None:
    baseline_dir = (
        tmp_path / "data" / "baselines" / "direct-cross-graph" / "algo3" / "evaluated"
    )
    llm_dir = tmp_path / "data" / "results" / "algo3" / "model-y" / "evaluated"
    baseline_path = baseline_dir / "baseline.csv"
    llm_path = llm_dir / "method3_results_evaluated_model_y.csv"
    output_path = tmp_path / "comparison.csv"

    baseline_dir.mkdir(parents=True, exist_ok=True)
    llm_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"Recall": [0.1, 0.3]}).to_csv(baseline_path, index=False)
    pd.DataFrame({"Recall": [0.4, 0.6]}).to_csv(llm_path, index=False)

    exit_code = main(
        [
            "analyze",
            "baseline-comparison",
            "--baseline-input",
            str(baseline_path),
            "--input",
            str(llm_path),
            "--metric",
            "Recall",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0

    actual = pd.read_csv(output_path)

    assert actual["algorithm"].tolist() == ["algo3"]
    assert actual["model"].tolist() == ["model-y"]
    assert actual["matched_file_count"].tolist() == [1]
    assert actual["baseline_mean"].round(6).tolist() == [0.2]
    assert actual["llm_mean"].round(6).tolist() == [0.5]
    assert actual["mean_delta"].round(6).tolist() == [0.3]


def test_cli_analyze_baseline_comparison_prefers_direct_cross_graph_baseline(tmp_path) -> None:
    baseline_dir = tmp_path / "data" / "baselines" / "direct-cross-graph" / "algo3" / "evaluated"
    llm_dir = tmp_path / "data" / "results" / "algo3" / "model-z" / "evaluated"
    llm_path = llm_dir / "method3_results_evaluated_model_z.csv"
    output_path = tmp_path / "comparison.csv"

    baseline_dir.mkdir(parents=True, exist_ok=True)
    llm_dir.mkdir(parents=True, exist_ok=True)

    pair_baseline_path = baseline_dir / "method3_results_evaluated_subgraph_1_to_subgraph_3.csv"
    aggregate_baseline_path = baseline_dir / "method3_results_evaluated_direct_cross_graph.csv"

    pd.DataFrame({"Recall": [0.8, 0.8]}).to_csv(pair_baseline_path, index=False)
    pd.DataFrame({"Recall": [0.1, 0.3]}).to_csv(aggregate_baseline_path, index=False)
    pd.DataFrame({"Recall": [0.4, 0.6]}).to_csv(llm_path, index=False)

    exit_code = main(
        [
            "analyze",
            "baseline-comparison",
            "--baseline-input",
            str(pair_baseline_path),
            "--baseline-input",
            str(aggregate_baseline_path),
            "--input",
            str(llm_path),
            "--metric",
            "Recall",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0

    actual = pd.read_csv(output_path)

    assert actual["baseline_mean"].round(6).tolist() == [0.2]
    assert actual["llm_mean"].round(6).tolist() == [0.5]
    assert actual["mean_delta"].round(6).tolist() == [0.3]
