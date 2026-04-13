from pathlib import Path

import pandas as pd
import pytest

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


def test_cli_analyze_hypothesis_bundle_writes_organized_review_artifacts(tmp_path) -> None:
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
            "hypothesis-bundle",
            "--results-root",
            str(results_root),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "bundle_manifest.csv").exists()
    assert (output_dir / "bundle_overview.csv").exists()
    assert (output_dir / "algo1" / "explanation" / "paired_tests.csv").exists()
    assert (output_dir / "algo2" / "convergence" / "factor_overview.csv").exists()
    assert (output_dir / "algo3" / "depth" / "significance_summary.csv").exists()


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


def test_cli_baseline_algo1_accepts_wordnet_and_edit_distance_strategies(tmp_path) -> None:
    wordnet_output_path = tmp_path / "algorithm1_wordnet_baseline.csv"
    edit_distance_output_path = tmp_path / "algorithm1_edit_distance_baseline.csv"

    wordnet_exit_code = main(
        [
            "baseline",
            "algo1",
            "--pair",
            "sg1_sg2",
            "--strategy",
            "wordnet-ontology-match",
            "--output",
            str(wordnet_output_path),
        ]
    )
    edit_distance_exit_code = main(
        [
            "baseline",
            "algo1",
            "--pair",
            "sg1_sg2",
            "--strategy",
            "edit-distance",
            "--output",
            str(edit_distance_output_path),
        ]
    )

    assert wordnet_exit_code == 0
    assert edit_distance_exit_code == 0
    assert wordnet_output_path.exists()
    assert edit_distance_output_path.exists()


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


def test_cli_analyze_stability_bundle_reorganizes_flat_files(tmp_path) -> None:
    # stability-bundle reads pre-computed flat stability CSVs, not raw/evaluated CSVs
    results_root = tmp_path / "results"
    results_root.mkdir(parents=True)
    output_dir = tmp_path / "bundle"

    _write_flat(
        results_root / "variability_incidence_by_algorithm.csv",
        [
            (
                "algorithm,metric,condition_count,varying_condition_count,"
                "varying_condition_share"
            ),
            "algo1,accuracy,576,2,0.0035",
            "algo3,Recall,96,44,0.4583",
        ],
    )
    _write_flat(
        results_root / "overall_metric_stability_by_algorithm.csv",
        [
            (
                "algorithm,metric,condition_count,mean_cv,median_cv,"
                "max_cv,mean_range_width,max_range_width"
            ),
            "algo1,accuracy,576,0.00002,0.0,0.012,0.00005,0.026",
            "algo3,Recall,96,3.21,3.87,3.87,0.337,1.0",
        ],
    )
    _write_flat(
        results_root / "algo2_convergence_stability_by_level.csv",
        [
            (
                "Convergence,metric,condition_count,mean_cv,median_cv,"
                "mean_range_width,max_range_width"
            ),
            "-1,accuracy,576,0.00003,0.0,0.00007,0.042",
            "1,accuracy,576,0.0,0.0,0.0,0.0",
        ],
    )
    _write_flat(
        results_root / "algo2_convergence_variability_incidence.csv",
        [
            (
                "Convergence,metric,condition_count,varying_condition_count,"
                "varying_condition_share"
            ),
            "-1,accuracy,576,1,0.0017",
            "1,accuracy,576,0,0.0",
        ],
    )

    exit_code = main(
        [
            "analyze",
            "stability-bundle",
            "--results-root",
            str(results_root),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "bundle_manifest.csv").exists()
    assert (output_dir / "bundle_overview.csv").exists()
    assert (output_dir / "variability_incidence_by_algorithm.csv").exists()
    assert (output_dir / "overall_metric_stability_by_algorithm.csv").exists()
    assert (output_dir / "algo2" / "convergence_stability_by_level.csv").exists()
    assert (output_dir / "algo2" / "convergence_variability_incidence.csv").exists()

    # Verify convergence=1 is perfectly stable (key finding)
    var = pd.read_csv(output_dir / "algo2" / "convergence_variability_incidence.csv")
    conv_1 = var[var["Convergence"] == 1].iloc[0]
    assert conv_1["varying_condition_count"] == 0
    assert conv_1["varying_condition_share"] == 0.0


def test_cli_analyze_figures_bundle_writes_distributional_summaries(tmp_path) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir(parents=True)
    output_dir = tmp_path / "bundle"

    _copy_fixture(
        "tests/reference_fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg1_sg2.csv",
        results_root / "algo1" / "gpt-5" / "evaluated" / "metrics_sg1_sg2.csv",
    )
    _copy_fixture(
        "tests/reference_fixtures/legacy/algo2/gpt-5/evaluated/metrics_sg1_sg2.csv",
        results_root / "algo2" / "gpt-5" / "evaluated" / "metrics_sg1_sg2.csv",
    )

    exit_code = main(
        [
            "analyze",
            "figures-bundle",
            "--results-root",
            str(results_root),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "bundle_manifest.csv").exists()
    assert (output_dir / "bundle_overview.csv").exists()
    assert (output_dir / "algo1_metric_rows.csv").exists()
    assert (output_dir / "algo2_metric_rows.csv").exists()
    assert (output_dir / "algo1" / "gpt-5" / "distributional_summary.csv").exists()
    assert (output_dir / "algo2" / "gpt-5" / "distributional_summary.csv").exists()

    overview = pd.read_csv(output_dir / "bundle_overview.csv")
    assert {"ci95_low", "ci95_high", "median", "q1", "q3"}.issubset(overview.columns)


def test_cli_analyze_replication_budget_supports_strict_ci_profile(tmp_path) -> None:
    input_path = tmp_path / "stability.csv"
    output_path = tmp_path / "budget.csv"
    _write_flat(
        input_path,
        [
            "metric,n,mean,sample_std",
            "accuracy,5,100,12",
        ],
    )

    exit_code = main(
        [
            "analyze",
            "replication-budget",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--ci-profile",
            "strict",
        ]
    )

    actual = pd.read_csv(output_path)

    assert exit_code == 0
    assert actual.iloc[0]["z_score"] == 1.96
    assert actual.iloc[0]["relative_half_width_target"] == 0.05
    assert actual.iloc[0]["required_total_runs"] == 23


def test_cli_analyze_replication_budget_supports_relaxed_ci_profile(tmp_path) -> None:
    input_path = tmp_path / "stability.csv"
    output_path = tmp_path / "budget.csv"
    _write_flat(
        input_path,
        [
            "metric,n,mean,sample_std",
            "accuracy,5,100,12",
        ],
    )

    exit_code = main(
        [
            "analyze",
            "replication-budget",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--ci-profile",
            "relaxed",
        ]
    )

    actual = pd.read_csv(output_path)

    assert exit_code == 0
    assert actual.iloc[0]["z_score"] == 1.645
    assert actual.iloc[0]["relative_half_width_target"] == 0.1
    assert actual.iloc[0]["required_total_runs"] == 5


def test_cli_analyze_plots_writes_revision_plot_family(tmp_path) -> None:
    results_root = tmp_path / "tracker"
    (results_root / "figure_exports").mkdir(parents=True)
    (results_root / "hypothesis_testing").mkdir(parents=True)
    (results_root / "output_variability").mkdir(parents=True)
    _write_flat(
        results_root / "figure_exports" / "bundle_overview.csv",
        [
            "algorithm,model,metric,mean,ci95_low,ci95_high,q1,q3",
            "algo1,gpt-5,accuracy,0.9,0.88,0.92,0.89,0.91",
            "algo3,gpt-5,Recall,0.1,0.02,0.18,0.0,0.15",
        ],
    )
    _write_flat(
        results_root / "hypothesis_testing" / "bundle_overview.csv",
        [
            "algorithm,factor,metric,mean_difference_average,significant_share",
            "algo1,Explanation,precision,0.02,0.5",
            "algo2,Convergence,accuracy,0.03,0.83",
        ],
    )
    _write_flat(
        results_root / "output_variability" / "bundle_overview.csv",
        [
            "algorithm,mean_pairwise_jaccard,breadth_expansion_ratio",
            "algo1,0.998,1.0",
            "algo3,0.077,4.13",
        ],
    )
    output_dir = tmp_path / "plots"

    exit_code = main(
        [
            "analyze",
            "plots",
            "--results-root",
            str(results_root),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "distribution_metrics.png").exists()
    assert (output_dir / "factor_effect_summary.png").exists()
    assert (output_dir / "raw_output_variability.png").exists()
    assert (output_dir / "main_metric_spread_boxplots.png").exists()
    assert (output_dir / "main_metric_spread_violins.png").exists()


def test_cli_run_paper_batch_writes_batch_summary(tmp_path) -> None:
    output_root = tmp_path / "runs"

    exit_code = main(
        [
            "run",
            "paper-batch",
            "--provider",
            "hf-transformers",
            "--model",
            "mistralai/Ministral-3-8B-Instruct-2512",
            "--embedding-model",
            "Qwen/Qwen3-Embedding-8B",
            "--output-root",
            str(output_root),
            "--replications",
            "1",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert (output_root / "batch_summary.csv").exists()


def test_cli_run_algo1_dry_run_limits_batch_to_requested_algorithm(tmp_path) -> None:
    output_root = tmp_path / "runs"

    exit_code = main(
        [
            "run",
            "algo1",
            "--provider",
            "hf-transformers",
            "--model",
            "mistralai/Ministral-3-8B-Instruct-2512",
            "--embedding-model",
            "Qwen/Qwen3-Embedding-8B",
            "--output-root",
            str(output_root),
            "--replications",
            "1",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    summary = pd.read_csv(output_root / "batch_summary.csv")
    assert summary["algorithm"].unique().tolist() == ["algo1"]


def test_cli_run_validate_config_writes_resolved_preview(tmp_path) -> None:
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        """
run:
  provider: hf-transformers
  output_root: /tmp/results
  replications: 5
runtime:
  seed: 7
  temperature: 0.0
  quantization: none
  device_policy: cuda-only
  thinking_mode_by_model:
    mistralai/Ministral-3-8B-Instruct-2512: acknowledged-unsupported
  context_policy:
    prompt_truncation: forbid
  max_new_tokens_by_schema:
    edge_list: 256
models:
  chat_models:
    - mistralai/Ministral-3-8B-Instruct-2512
  embedding_model: Qwen/Qwen3-Embedding-8B
decoding:
  greedy:
    enabled: true
inputs:
  graph_source: default
shared_fragments:
  assistant_role: "You are a helpful assistant."
algorithms:
  algo1:
    base_fragments: [assistant_role]
    factors: {}
    fragment_definitions: {}
    prompt_templates:
      body: "Task body."
""",
        encoding="utf-8",
    )
    output_dir = tmp_path / "preview"

    exit_code = main(
        [
            "run",
            "validate-config",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "resolved_run_config.yaml").exists()
    assert (output_dir / "resolved_run_plan.json").exists()
    assert (output_dir / "prompt_preview" / "algo1" / "base.txt").exists()


def test_cli_run_status_reports_batch_health_as_json(tmp_path, capsys) -> None:
    results_root = tmp_path / "results"
    run_dir = (
        results_root / "runs" / "algo1" / "model" / "greedy" / "sg1_sg2" / "00000" / "rep_00"
    )
    run_dir.mkdir(
        parents=True, exist_ok=True
    )
    (run_dir / "state.json").write_text(
        '{"status": "finished"}',
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        '{"status": "finished"}',
        encoding="utf-8",
    )

    exit_code = main(
        [
            "run",
            "status",
            "--results-root",
            str(results_root),
            "--json",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"finished_count": 1' in captured.out
    assert '"failed_count": 0' in captured.out


def test_cli_run_refresh_ledger_reports_updated_counts_as_json(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    results_root = tmp_path / "results"
    ledger_root = results_root / "hf-paper-batch-canonical"
    ledger_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.refresh_ledger",
        lambda *, results_root, ledger_root: {
            "expected_total_runs": 2,
            "finished_count": 1,
            "retryable_failed_count": 0,
            "terminal_failed_count": 0,
            "pending_count": 1,
            "records": [],
        },
    )

    exit_code = main(
        [
            "run",
            "refresh-ledger",
            "--results-root",
            str(results_root),
            "--ledger-root",
            str(ledger_root),
            "--json",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"finished_count": 1' in captured.out
    assert '"pending_count": 1' in captured.out


def test_cli_run_resume_preflight_reports_local_resume_readiness(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run: {}\n", encoding="utf-8")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    results_root = tmp_path / "results"
    results_root.mkdir()

    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.load_hf_run_config",
        lambda _path: object(),
    )
    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.build_resume_preflight_report",
        lambda *, config, repo_root, results_root, allow_empty=False: {
            "config_loaded": config is not None,
            "repo_root": str(repo_root),
            "results_root": str(results_root),
            "pending_count": 12,
            "can_resume": True,
            "allow_empty": allow_empty,
        },
    )

    exit_code = main(
        [
            "run",
            "resume-preflight",
            "--config",
            str(config_path),
            "--repo-root",
            str(repo_root),
            "--results-root",
            str(results_root),
            "--json",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"pending_count": 12' in captured.out
    assert '"can_resume": true' in captured.out


def test_cli_run_resume_sweep_reports_root_classification_as_json(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    results_root = tmp_path / "results"
    results_root.mkdir()

    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.build_resume_sweep_report",
        lambda *, repo_root, results_root: {
            "repo_root": str(repo_root),
            "results_root": str(results_root),
            "root_count": 2,
            "ready_count": 1,
            "needs_config_fix_count": 1,
            "invalid_config_count": 1,
            "active_count": 0,
            "finished_count": 0,
            "roots": [
                {
                    "results_root": str(Path(results_root) / "hf-paper-batch-algo1-olmo-current"),
                    "classification": "resume-ready",
                },
                {
                    "results_root": str(Path(results_root) / "hf-paper-batch-algo1-qwen"),
                    "classification": "needs-config-fix",
                },
            ],
        },
    )

    exit_code = main(
        [
            "run",
            "resume-sweep",
            "--repo-root",
            str(repo_root),
            "--results-root",
            str(results_root),
            "--json",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"ready_count": 1' in captured.out
    assert '"needs_config_fix_count": 1' in captured.out
    assert '"invalid_config_count": 1' in captured.out
    assert '"classification": "resume-ready"' in captured.out


def test_cli_run_prefetch_runtime_reports_prefetched_models(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run: {}\n", encoding="utf-8")

    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.load_hf_run_config",
        lambda _path: object(),
    )
    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.prefetch_runtime_for_config",
        lambda *, config: {
            "chat_models": ["Qwen/Qwen3.5-9B"],
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        },
    )

    exit_code = main(
        [
            "run",
            "prefetch-runtime",
            "--config",
            str(config_path),
            "--json",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"chat_models": [' in captured.out
    assert '"embedding_model": "Qwen/Qwen3-Embedding-0.6B"' in captured.out


def test_cli_run_prefetch_runtime_reports_plain_text_models(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run: {}\n", encoding="utf-8")

    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.load_hf_run_config",
        lambda _path: object(),
    )
    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.prefetch_runtime_for_config",
        lambda *, config: {
            "chat_models": ["Qwen/Qwen3.5-9B", "allenai/Olmo-3-7B-Instruct"],
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        },
    )

    exit_code = main(
        [
            "run",
            "prefetch-runtime",
            "--config",
            str(config_path),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "chat_models=Qwen/Qwen3.5-9B,allenai/Olmo-3-7B-Instruct" in captured.out
    assert "embedding_model=Qwen/Qwen3-Embedding-0.6B" in captured.out


def test_cli_run_prefetch_runtime_rejects_non_list_chat_models(
    monkeypatch,
    tmp_path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run: {}\n", encoding="utf-8")

    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.load_hf_run_config",
        lambda _path: object(),
    )
    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.prefetch_runtime_for_config",
        lambda *, config: {
            "chat_models": "Qwen/Qwen3.5-9B",
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        },
    )

    with pytest.raises(
        ValueError,
        match="Prefetch runtime report chat_models must be a list of strings",
    ):
        main(
            [
                "run",
                "prefetch-runtime",
                "--config",
                str(config_path),
            ]
        )


def test_cli_run_status_reports_active_worker_fields(monkeypatch, tmp_path, capsys) -> None:
    results_root = tmp_path / "results"
    run_dir = (
        results_root / "runs" / "algo1" / "model" / "greedy" / "sg1_sg2" / "00000" / "rep_00"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "state.json").write_text('{"status": "running"}', encoding="utf-8")
    (run_dir / "worker_state.json").write_text(
        '{"status": "running", "pid": 4242, "model_loaded": true}',
        encoding="utf-8",
    )
    (run_dir / "active_stage.json").write_text(
        '{"status": "running", "schema_name": "edge_list"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "llm_conceptual_modeling.hf_batch.monitoring._query_gpu_processes",
        lambda: [{"pid": 4242, "used_gpu_memory_mib": 1234}],
    )

    exit_code = main(
        [
            "run",
            "status",
            "--results-root",
            str(results_root),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "running=1" in captured.out
    assert "worker_pid=4242" in captured.out
    assert "worker_status=running" in captured.out
    assert "active_stage_age_seconds=" in captured.out


def test_cli_run_drain_remaining_reports_queue_as_json(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    results_root = tmp_path / "results"
    results_root.mkdir()
    state_path = tmp_path / "drain-state.json"

    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.build_drain_plan",
        lambda **_kwargs: {
            "queue": [
                {
                    "results_root": str(results_root / "hf-paper-batch-algo2-olmo-current"),
                    "phase": "safe",
                    "profile_name": "olmo-safe",
                }
            ],
            "safe_queue_count": 1,
            "risky_queue_count": 0,
            "adopted_results_root": str(results_root / "hf-paper-batch-algo2-olmo-current"),
            "state_file": str(state_path),
        },
    )

    exit_code = main(
        [
            "run",
            "drain-remaining",
            "--repo-root",
            str(repo_root),
            "--results-root",
            str(results_root),
            "--ssh-command",
            "ssh -p 2222 root@example.com",
            "--state-file",
            str(state_path),
            "--plan-only",
            "--json",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"safe_queue_count": 1' in captured.out
    assert '"profile_name": "olmo-safe"' in captured.out


def test_cli_run_drain_status_reports_saved_state(monkeypatch, tmp_path, capsys) -> None:
    state_path = tmp_path / "drain-state.json"
    state_path.write_text(
        """
        {
          "health": "healthy",
          "current_phase": "safe",
          "current_results_root": "/tmp/results/hf-paper-batch-algo2-olmo-current",
          "current_status": {"finished_count": 10, "pending_count": 5, "failed_count": 1}
        }
        """,
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.read_drain_state_report",
        lambda state_file: {
            "state_file": str(state_file),
            "health": "healthy",
            "current_phase": "safe",
            "current_results_root": "/tmp/results/hf-paper-batch-algo2-olmo-current",
            "current_status": {"finished_count": 10, "pending_count": 5, "failed_count": 1},
        },
    )

    exit_code = main(
        [
            "run",
            "drain-status",
            "--state-file",
            str(state_path),
            "--json",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"health": "healthy"' in captured.out
    assert '"current_phase": "safe"' in captured.out


def test_cli_run_smoke_executes_single_selected_spec(monkeypatch, tmp_path, capsys) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
run:
  provider: hf-transformers
  output_root: /tmp/hf-smoke
  replications: 1
runtime:
  seed: 123
  temperature: 0.0
  quantization: none
  device_policy: cuda-only
  thinking_mode_by_model:
    Qwen/Qwen3.5-9B: disabled
  context_policy:
    prompt_truncation: forbid
    safety_margin_tokens: 64
  max_new_tokens_by_schema:
    edge_list: 256
    vote_list: 64
    label_list: 128
    children_by_label: 384
models:
  chat_models:
    - Qwen/Qwen3.5-9B
  embedding_model: Qwen/Qwen3-Embedding-0.6B
decoding:
  greedy:
    enabled: true
inputs:
  graph_source: default
shared_fragments:
  assistant_role: "You are a helpful assistant."
algorithms:
  algo1:
    pair_names: [sg1_sg2, sg2_sg3, sg3_sg1]
    base_fragments: [assistant_role]
    factors:
      explanation:
        column: Explanation
        levels: [-1, 1]
        runtime_field: include_explanation
        low_runtime_value: false
        high_runtime_value: true
        low_fragments: []
        high_fragments: []
      example:
        column: Example
        levels: [-1, 1]
        runtime_field: include_example
        low_runtime_value: false
        high_runtime_value: true
        low_fragments: []
        high_fragments: []
      counterexample:
        column: Counterexample
        levels: [-1, 1]
        runtime_field: include_counterexample
        low_runtime_value: false
        high_runtime_value: true
        low_fragments: []
        high_fragments: []
      array_repr:
        column: Array/List(1/-1)
        levels: [-1, 1]
        runtime_field: use_array_representation
        low_runtime_value: false
        high_runtime_value: true
        low_fragments: []
        high_fragments: []
      adjacency_repr:
        column: Tag/Adjacency(1/-1)
        levels: [-1, 1]
        runtime_field: use_adjacency_notation
        low_runtime_value: false
        high_runtime_value: true
        low_fragments: []
        high_fragments: []
    fragment_definitions: {}
    prompt_templates:
      direct_edge: "Task body."
      cove_verification: "Verify body."
""",
        encoding="utf-8",
    )
    captured_call: dict[str, object] = {}

    def fake_run_single_spec(**kwargs):
        captured_call.update(kwargs)
        return {"pair_name": "sg2_sg3"}

    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.run_single_spec",
        fake_run_single_spec,
    )

    exit_code = main(
        [
            "run",
            "smoke",
            "--config",
            str(config_path),
            "--algorithm",
            "algo1",
            "--model",
            "Qwen/Qwen3.5-9B",
            "--pair-name",
            "sg2_sg3",
            "--condition-bits",
            "00000",
            "--decoding",
            "greedy",
            "--output-root",
            str(tmp_path / "smoke"),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured_call["spec"].pair_name == "sg2_sg3"
    assert captured_call["spec"].condition_bits == "00000"
    assert captured_call["spec"].model == "Qwen/Qwen3.5-9B"
    assert captured_call["spec"].decoding.algorithm == "greedy"
    assert '"pair_name": "sg2_sg3"' in captured.out


def _write_flat(path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
