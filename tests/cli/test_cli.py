import json
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from llm_conceptual_modeling.cli import main
from llm_conceptual_modeling.commands.run import (
    _decoding_from_args,
    _drain_remaining_report,
    _handle_drain_remaining,
    _handle_drain_status,
    _handle_experiment_run,
    _handle_prefetch_runtime,
    _handle_prepare_qwen_algo1_tail,
    _handle_qwen_algo1_tail_preflight,
    _handle_refresh_ledger,
    _handle_resume_preflight,
    _handle_resume_sweep,
    _handle_smoke,
    _handle_status,
    _handle_write_unfinished_manifest,
    _load_optional_run_config,
    _prefetch_runtime_report_lines,
    _require_hf_transformers_provider,
    _resume_preflight_output_key,
    _resume_sweep_output_key,
    _run_algorithms,
    _status_output_key,
    prefetch_runtime_for_config,
)


def test_cli_main_is_implemented_in_the_commands_package() -> None:
    assert main.__module__ == "llm_conceptual_modeling.commands.cli"


def test_cli_shim_exports_script_entrypoint() -> None:
    from llm_conceptual_modeling.cli import run

    assert run.__module__ == "llm_conceptual_modeling.commands.cli"


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


def test_cli_analyze_map_extension_baseline_bundle_writes_outputs(tmp_path) -> None:
    results_root = tmp_path / "hf-map-extension-canonical"
    output_dir = tmp_path / "bundle"
    raw_row_path = (
        results_root
        / "runs"
        / "algo3"
        / "Qwen__Qwen3.5-9B"
        / "greedy"
        / "case_a"
        / "subgraph_1_to_subgraph_2"
        / "000"
        / "rep_00"
        / "raw_row.json"
    )
    raw_row_path.parent.mkdir(parents=True, exist_ok=True)
    raw_row_path.write_text(
        json.dumps(
            {
                "Source Graph": "[('a', 'b')]",
                "Target Graph": "[('x', 'y')]",
                "Mother Graph": "[('a', 'b'), ('x', 'y'), ('b', 'x')]",
                "Results": "[('b', 'bridge'), ('bridge', 'x')]",
                "Recall": 1.0,
                "model": "Qwen/Qwen3.5-9B",
                "graph_source": "case_a",
                "pair_name": "subgraph_1_to_subgraph_2",
                "decoding_algorithm": "greedy",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame.from_records(
        [
            {
                "algorithm": "algo3",
                "model": "Qwen/Qwen3.5-9B",
                "graph_source": "case_a",
                "decoding_algorithm": "greedy",
                "pair_name": "subgraph_1_to_subgraph_2",
                "replication": 0,
                "status": "finished",
                "raw_row_path": str(raw_row_path),
            }
        ]
    ).to_csv(results_root / "batch_summary.csv", index=False)

    exit_code = main(
        [
            "analyze",
            "map-extension-baseline-bundle",
            "--results-root",
            str(results_root),
            "--output-dir",
            str(output_dir),
            "--random-repetitions",
            "5",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "row_level_baseline_comparison.csv").exists()
    assert (output_dir / "map_extension_model_vs_baseline.csv").exists()


def test_cli_analyze_variance_decomposition_bundle_writes_bundle_outputs(tmp_path) -> None:
    results_root = Path("data/results/open_weights/hf-paper-batch-canonical")
    output_dir = tmp_path / "variance_decomposition"

    exit_code = main(
        [
            "analyze",
            "variance-decomposition-bundle",
            "--results-root",
            str(results_root),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "variance_decomposition.csv").exists()
    assert (output_dir / "variance_decomposition_algo1.csv").exists()
    assert (output_dir / "variance_decomposition_algo1.tex").exists()
    assert (output_dir / "variance_decomposition.tex").exists()
    assert (output_dir / "README.md").exists()
    decomposition = pd.read_csv(output_dir / "variance_decomposition.csv")
    assert {"algo1", "algo2", "algo3"}.issubset(set(decomposition["algorithm"]))


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
            ("algorithm,metric,condition_count,varying_condition_count,varying_condition_share"),
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
            ("Convergence,metric,condition_count,varying_condition_count,varying_condition_share"),
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


def test_cli_analyze_replication_budget_sufficiency_writes_summary(tmp_path) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    (results_root / "ledger.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "identity": {
                            "algorithm": "algo1",
                            "model": "Qwen/Qwen3.5-9B",
                            "condition_label": "greedy",
                            "pair_name": "sg1_sg2",
                            "condition_bits": "00000",
                            "replication": 0,
                        },
                        "status": "finished",
                        "winner": {
                            "metrics": {"accuracy": 100.0},
                            "status": "finished",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "summary.csv"

    exit_code = main(
        [
            "analyze",
            "replication-budget-sufficiency",
            "--results-root",
            str(results_root),
            "--output",
            str(output_path),
        ]
    )

    actual = pd.read_csv(output_path)

    assert exit_code == 0
    assert set(actual["profile"]) == {"ci95_rel05", "ci90_rel05"}
    assert actual.iloc[0]["source_finished_run_count"] == 1


def test_cli_analyze_replication_budget_sufficiency_writes_compact_table(
    tmp_path,
) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    (results_root / "ledger.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "identity": {
                            "algorithm": "algo1",
                            "model": "Qwen/Qwen3.5-9B",
                            "condition_label": "greedy",
                            "pair_name": "sg1_sg2",
                            "condition_bits": "00000",
                            "replication": replication,
                        },
                        "status": "finished",
                        "winner": {
                            "metrics": {"accuracy": 100.0},
                            "status": "finished",
                        },
                    }
                    for replication in range(5)
                ]
            }
        ),
        encoding="utf-8",
    )
    detailed_path = tmp_path / "detailed.csv"
    compact_path = tmp_path / "compact.csv"

    exit_code = main(
        [
            "analyze",
            "replication-budget-sufficiency",
            "--results-root",
            str(results_root),
            "--output",
            str(detailed_path),
            "--compact-output",
            str(compact_path),
        ]
    )

    compact = pd.read_csv(compact_path)

    assert exit_code == 0
    assert compact.columns.tolist()[:4] == [
        "algorithm",
        "decoding",
        "qwen_runs",
        "qwen_condition_metrics",
    ]
    assert compact.iloc[0]["qwen_runs"] == 5


def test_cli_analyze_replication_budget_sufficiency_can_include_graph_source_in_compact_output(
    tmp_path,
) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    (results_root / "ledger.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "identity": {
                            "algorithm": "algo3",
                            "model": "Qwen/Qwen3.5-9B",
                            "condition_label": "beam_num_beams_6",
                            "graph_source": "babs_johnson",
                            "pair_name": "subgraph_1_to_subgraph_3",
                            "condition_bits": "000",
                            "replication": replication,
                        },
                        "status": "finished",
                        "winner": {
                            "metrics": {"recall": value},
                            "status": "finished",
                        },
                    }
                    for replication, value in enumerate([20.0, 20.0, 50.0, 80.0, 80.0])
                ]
            }
        ),
        encoding="utf-8",
    )
    detailed_path = tmp_path / "detailed.csv"
    compact_path = tmp_path / "compact.csv"

    exit_code = main(
        [
            "analyze",
            "replication-budget-sufficiency",
            "--results-root",
            str(results_root),
            "--output",
            str(detailed_path),
            "--compact-output",
            str(compact_path),
            "--include-graph-source",
        ]
    )

    compact = pd.read_csv(compact_path)

    assert exit_code == 0
    assert compact.columns.tolist()[:3] == ["algorithm", "graph_source", "decoding"]
    assert compact.iloc[0]["graph_source"] == "babs_johnson"


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
    run_dir = results_root / "runs" / "algo1" / "model" / "greedy" / "sg1_sg2" / "00000" / "rep_00"
    run_dir.mkdir(parents=True, exist_ok=True)
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


def test_cli_run_write_unfinished_manifest_reports_identity_count(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    ledger_root = tmp_path / "ledger"
    ledger_root.mkdir()
    manifest_path = tmp_path / "shard_manifest.json"

    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.write_unfinished_shard_manifest",
        lambda *, results_root, ledger_root, manifest_path: {
            "generated_at": "2026-04-21T10:00:00+00:00",
            "results_root": str(results_root),
            "ledger_root": str(ledger_root),
            "active_chat_models": ["Qwen/Qwen3.5-9B"],
            "shard_count": 1,
            "shard_index": 0,
            "identities": [
                {
                    "algorithm": "algo1",
                    "condition_bits": "00000",
                    "condition_label": "greedy",
                    "model": "Qwen/Qwen3.5-9B",
                    "pair_name": "sg1_sg2",
                    "replication": 0,
                }
            ],
        },
    )

    exit_code = main(
        [
            "run",
            "write-unfinished-manifest",
            "--results-root",
            str(results_root),
            "--ledger-root",
            str(ledger_root),
            "--manifest-path",
            str(manifest_path),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert f"manifest_path={manifest_path}" in captured.out
    assert "identity_count=1" in captured.out
    assert "active_chat_models=Qwen/Qwen3.5-9B" in captured.out


def test_cli_run_qwen_algo1_tail_preflight_reports_nested_resume_counts(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    canonical_results_root = tmp_path / "canonical"
    canonical_results_root.mkdir()
    tail_results_root = tmp_path / "tail"
    tail_results_root.mkdir()

    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.build_qwen_algo1_tail_preflight_report",
        lambda *, repo_root, canonical_results_root, tail_results_root, watcher_status_path=None: {
            "repo_root": str(repo_root),
            "canonical_results_root": str(canonical_results_root),
            "tail_results_root": str(tail_results_root),
            "watcher_status": None,
            "canonical_ledger": {"finished_count": 0},
            "tail_ledger": {"pending_count": 10},
            "resume_preflight": {
                "results_root": str(tail_results_root),
                "total_runs": 10,
                "finished_count": 0,
                "failed_count": 0,
                "pending_count": 10,
                "running_count": 0,
                "can_resume": True,
                "resume_mode": "resume",
            },
        },
    )

    exit_code = main(
        [
            "run",
            "qwen-algo1-tail-preflight",
            "--repo-root",
            str(repo_root),
            "--canonical-results-root",
            str(canonical_results_root),
            "--tail-results-root",
            str(tail_results_root),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert f"tail_results_root={tail_results_root}" in captured.out
    assert "tail_pending_count=10" in captured.out
    assert "tail_can_resume=True" in captured.out


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
    run_dir = results_root / "runs" / "algo1" / "model" / "greedy" / "sg1_sg2" / "00000" / "rep_00"
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


def test_plain_run_status_handlers_preserve_scriptable_output(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    results_root = tmp_path / "results"
    ledger_root = tmp_path / "ledger"
    state_path = tmp_path / "state.json"
    config_path = tmp_path / "config.yaml"
    repo_root = tmp_path / "repo"
    for path in (results_root, ledger_root, repo_root):
        path.mkdir()
    config_path.write_text("run: {}\n", encoding="utf-8")

    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.refresh_ledger",
        lambda **_kwargs: {
            "expected_total_runs": 2,
            "finished_count": 1,
            "retryable_failed_count": 0,
            "terminal_failed_count": 0,
            "pending_count": 1,
        },
    )
    assert (
        _handle_refresh_ledger(
            Namespace(results_root=results_root, ledger_root=ledger_root, json=False)
        )
        == 0
    )
    assert f"ledger_root={ledger_root}" in capsys.readouterr().out

    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.load_hf_run_config",
        lambda _path: object(),
    )
    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.build_resume_preflight_report",
        lambda **_kwargs: {
            "results_root": str(results_root),
            "total_runs": 12,
            "finished_count": 4,
            "failed_count": 1,
            "pending_count": 7,
            "can_resume": True,
            "resume_mode": "resume",
        },
    )
    assert (
        _handle_resume_preflight(
            Namespace(
                config=config_path,
                repo_root=repo_root,
                results_root=results_root,
                allow_empty=False,
                json=False,
            )
        )
        == 0
    )
    preflight_output = capsys.readouterr().out
    assert "finished=4" in preflight_output
    assert "pending=7" in preflight_output

    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.build_resume_sweep_report",
        lambda **_kwargs: {
            "repo_root": str(repo_root),
            "results_root": str(results_root),
            "root_count": 2,
            "ready_count": 1,
            "needs_config_fix_count": 0,
            "invalid_config_count": 0,
            "active_count": 1,
            "finished_count": 0,
        },
    )
    assert (
        _handle_resume_sweep(Namespace(repo_root=repo_root, results_root=results_root, json=False))
        == 0
    )
    assert "roots=2" in capsys.readouterr().out

    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run._drain_remaining_report",
        lambda _args: {
            "state_file": str(state_path),
            "safe_queue_count": 2,
            "risky_queue_count": 1,
            "adopted_results_root": str(results_root / "active"),
        },
    )
    assert _handle_drain_remaining(Namespace(json=False)) == 0
    drain_output = capsys.readouterr().out
    assert "safe_queue_count=2" in drain_output
    assert "adopted_results_root=" in drain_output

    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.read_drain_state_report",
        lambda _state_file: {
            "health": "healthy",
            "current_phase": "safe",
            "current_results_root": str(results_root),
        },
    )
    assert _handle_drain_status(Namespace(state_file=state_path, json=False)) == 0
    status_output = capsys.readouterr().out
    assert "health=healthy" in status_output
    assert "current_phase=safe" in status_output


def test_prepare_qwen_algo1_tail_handler_preserves_plain_and_json_output(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    report = {
        "tail_results_root": str(tmp_path / "tail"),
        "identity_count": 10,
        "config_path": str(tmp_path / "tail" / "runtime_config.yaml"),
        "manifest_path": str(tmp_path / "tail" / "shard_manifest.json"),
    }
    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.prepare_qwen_algo1_tail_bundle",
        lambda **_kwargs: report,
    )
    args = Namespace(
        canonical_results_root=tmp_path / "canonical",
        tail_results_root=tmp_path / "tail",
        remote_output_root="/workspace/results/tail",
        json=False,
    )

    assert _handle_prepare_qwen_algo1_tail(args) == 0
    plain_output = capsys.readouterr().out
    assert "identity_count=10" in plain_output
    assert "manifest_path=" in plain_output

    args.json = True
    assert _handle_prepare_qwen_algo1_tail(args) == 0
    assert json.loads(capsys.readouterr().out) == report


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


def test_cli_run_smoke_passes_graph_source_to_selected_spec(monkeypatch, tmp_path, capsys) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        Path("configs/hf_transformers_open_weight_map_extension.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    captured_call: dict[str, object] = {}

    def fake_run_single_spec(**kwargs):
        captured_call.update(kwargs)
        return {"graph_source": kwargs["spec"].graph_source}

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
            "algo3",
            "--model",
            "Qwen/Qwen3.5-9B",
            "--graph-source",
            "clarice_starling",
            "--pair-name",
            "subgraph_1_to_subgraph_3",
            "--condition-bits",
            "000",
            "--decoding",
            "beam",
            "--num-beams",
            "6",
            "--output-root",
            str(tmp_path / "smoke"),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured_call["spec"].graph_source == "clarice_starling"
    assert '"graph_source": "clarice_starling"' in captured.out


@pytest.mark.parametrize(
    ("converter", "key", "expected"),
    [
        (_resume_preflight_output_key, "finished_count", "finished"),
        (_resume_preflight_output_key, "failed_count", "failed"),
        (_resume_preflight_output_key, "pending_count", "pending"),
        (_resume_sweep_output_key, "root_count", "roots"),
        (_resume_sweep_output_key, "ready_count", "ready"),
        (_resume_sweep_output_key, "needs_config_fix_count", "needs_config_fix"),
        (_resume_sweep_output_key, "invalid_config_count", "invalid_config"),
        (_resume_sweep_output_key, "active_count", "active"),
        (_resume_sweep_output_key, "finished_count", "finished"),
        (_status_output_key, "total_runs", "total"),
        (_status_output_key, "finished_count", "finished"),
        (_status_output_key, "failed_count", "failed"),
        (_status_output_key, "running_count", "running"),
        (_status_output_key, "pending_count", "pending"),
    ],
)
def test_run_plain_output_key_aliases_are_stable(converter, key, expected) -> None:
    assert converter(key) == expected


@pytest.mark.parametrize(
    "converter",
    [_resume_preflight_output_key, _resume_sweep_output_key, _status_output_key],
)
def test_run_plain_output_key_aliases_preserve_unknown_keys(converter) -> None:
    assert converter("unmapped_field") == "unmapped_field"


def test_resume_preflight_handler_forwards_inputs_and_preserves_json_contract(
    monkeypatch, tmp_path, capsys
) -> None:
    config_path = tmp_path / "config.yaml"
    repo_root = tmp_path / "repo"
    results_root = tmp_path / "results"
    config_marker = object()
    loaded_paths = []
    forwarded = {}
    report = {
        "results_root": str(results_root),
        "allow_empty": True,
        "config_is_expected": True,
        "repo_root": str(repo_root),
    }

    def fake_load(path):
        loaded_paths.append(path)
        return config_marker

    def fake_build(*, config, repo_root, results_root, allow_empty):
        forwarded.update(
            config=config,
            repo_root=repo_root,
            results_root=results_root,
            allow_empty=allow_empty,
        )
        return {
            **report,
            "config_is_expected": config is config_marker,
        }

    monkeypatch.setattr("llm_conceptual_modeling.commands.run.load_hf_run_config", fake_load)
    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.build_resume_preflight_report",
        fake_build,
    )

    args = Namespace(
        config=config_path,
        repo_root=repo_root,
        results_root=results_root,
        allow_empty=True,
        json=True,
    )

    assert _handle_resume_preflight(args) == 0
    assert loaded_paths == [config_path]
    assert forwarded == {
        "config": config_marker,
        "repo_root": repo_root,
        "results_root": results_root,
        "allow_empty": True,
    }
    assert capsys.readouterr().out == json.dumps(report, indent=2, sort_keys=True) + "\n"


def test_resume_sweep_handler_forwards_inputs_and_preserves_plain_and_json_contracts(
    monkeypatch, tmp_path, capsys
) -> None:
    repo_root = tmp_path / "repo"
    results_root = tmp_path / "results"
    forwarded = {}
    report = {
        "repo_root": str(repo_root),
        "results_root": str(results_root),
        "root_count": 2,
        "ready_count": 1,
        "needs_config_fix_count": 3,
        "invalid_config_count": 4,
        "active_count": 5,
        "finished_count": 6,
    }

    def fake_build(*, repo_root, results_root):
        forwarded.update(repo_root=repo_root, results_root=results_root)
        return report

    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.build_resume_sweep_report",
        fake_build,
    )
    args = Namespace(repo_root=repo_root, results_root=results_root, json=False)

    assert _handle_resume_sweep(args) == 0
    assert forwarded == {"repo_root": repo_root, "results_root": results_root}
    assert capsys.readouterr().out == (
        f"repo_root={repo_root}\n"
        f"results_root={results_root}\n"
        "roots=2\n"
        "ready=1\n"
        "needs_config_fix=3\n"
        "invalid_config=4\n"
        "active=5\n"
        "finished=6\n"
    )

    args.json = True
    assert _handle_resume_sweep(args) == 0
    assert capsys.readouterr().out == json.dumps(report, indent=2, sort_keys=True) + "\n"


def test_prefetch_runtime_handler_forwards_config_and_preserves_json_contract(
    monkeypatch, tmp_path, capsys
) -> None:
    config_path = tmp_path / "config.yaml"
    config_marker = object()
    loaded_paths = []
    forwarded = []
    report = {
        "embedding_model": "embedding-model",
        "chat_models": ["model-a", "model-b"],
    }

    def fake_load(path):
        loaded_paths.append(path)
        return config_marker

    def fake_prefetch(*, config):
        forwarded.append(config)
        return report

    monkeypatch.setattr("llm_conceptual_modeling.commands.run.load_hf_run_config", fake_load)
    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.prefetch_runtime_for_config",
        fake_prefetch,
    )

    assert _handle_prefetch_runtime(Namespace(config=config_path, json=True)) == 0
    assert loaded_paths == [config_path]
    assert forwarded == [config_marker]
    assert capsys.readouterr().out == json.dumps(report, indent=2, sort_keys=True) + "\n"


def test_prefetch_runtime_report_lines_validate_contract_exactly() -> None:
    assert _prefetch_runtime_report_lines(
        {"chat_models": ["model-a", "model-b"], "embedding_model": "embedding-model"}
    ) == [
        "chat_models=model-a,model-b",
        "embedding_model=embedding-model",
    ]

    with pytest.raises(ValueError) as exc_info:
        _prefetch_runtime_report_lines(
            {"chat_models": "model-a", "embedding_model": "embedding-model"}
        )
    assert str(exc_info.value) == "Prefetch runtime report chat_models must be a list of strings"

    with pytest.raises(ValueError) as exc_info:
        _prefetch_runtime_report_lines(
            {"chat_models": ["model-a", 7], "embedding_model": "embedding-model"}
        )
    assert str(exc_info.value) == "Prefetch runtime report chat_models must be a list of strings"

    with pytest.raises(ValueError) as exc_info:
        _prefetch_runtime_report_lines({"chat_models": [], "embedding_model": 7})
    assert str(exc_info.value) == "Prefetch runtime report embedding_model must be a string"


def test_status_handler_preserves_exact_json_and_plain_contracts(
    monkeypatch, tmp_path, capsys
) -> None:
    results_root = tmp_path / "results"
    json_status = {
        "total_runs": 10,
        "running_count": 1,
        "percent_complete": 37.5,
        "pending_count": 4,
        "finished_count": 3,
        "failed_count": 2,
    }
    plain_status = {
        "failed_count": 2,
        "finished_count": 3,
        "pending_count": 4,
        "percent_complete": 37.5,
        "running_count": 1,
        "total_runs": 10,
    }
    statuses = [json_status, plain_status]
    received_roots = []

    def fake_collect(root):
        received_roots.append(root)
        return statuses.pop(0)

    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.collect_batch_status",
        fake_collect,
    )

    assert _handle_status(Namespace(results_root=results_root, json=True)) == 0
    assert capsys.readouterr().out == json.dumps(json_status, indent=2, sort_keys=True) + "\n"

    assert _handle_status(Namespace(results_root=results_root, json=False)) == 0
    assert capsys.readouterr().out == (
        "total=10\nfinished=3\nfailed=2\nrunning=1\npending=4\ncomplete=37.5%\n"
    )
    assert received_roots == [results_root, results_root]


def test_refresh_ledger_handler_forwards_paths_and_preserves_exact_contract(
    monkeypatch, tmp_path, capsys
) -> None:
    results_root = tmp_path / "results"
    ledger_root = tmp_path / "ledger"
    ledger = {
        "expected_total_runs": 10,
        "finished_count": 3,
        "retryable_failed_count": 1,
        "terminal_failed_count": 2,
        "pending_count": 4,
    }
    forwarded = []

    def fake_refresh(*, results_root, ledger_root):
        forwarded.append((results_root, ledger_root))
        return ledger

    monkeypatch.setattr("llm_conceptual_modeling.commands.run.refresh_ledger", fake_refresh)
    args = Namespace(results_root=results_root, ledger_root=ledger_root, json=False)

    assert _handle_refresh_ledger(args) == 0
    assert forwarded == [(results_root, ledger_root)]
    assert capsys.readouterr().out == (
        f"ledger_root={ledger_root}\n"
        "expected_total_runs=10\n"
        "finished_count=3\n"
        "retryable_failed_count=1\n"
        "terminal_failed_count=2\n"
        "pending_count=4\n"
    )

    args.json = True
    assert _handle_refresh_ledger(args) == 0
    assert capsys.readouterr().out == json.dumps(ledger, indent=2, sort_keys=True) + "\n"


def test_write_unfinished_manifest_handler_forwards_paths_and_json_contract(
    monkeypatch, tmp_path, capsys
) -> None:
    results_root = tmp_path / "results"
    ledger_root = tmp_path / "ledger"
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "manifest_path": str(manifest_path),
        "identities": [{"algorithm": "algo1"}],
        "active_chat_models": ["model-a", "model-b"],
    }
    forwarded = []

    def fake_write(*, results_root, ledger_root, manifest_path):
        forwarded.append((results_root, ledger_root, manifest_path))
        return manifest

    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.write_unfinished_shard_manifest",
        fake_write,
    )

    assert (
        _handle_write_unfinished_manifest(
            Namespace(
                results_root=results_root,
                ledger_root=ledger_root,
                manifest_path=manifest_path,
                json=True,
            )
        )
        == 0
    )
    assert forwarded == [(results_root, ledger_root, manifest_path)]
    assert capsys.readouterr().out == json.dumps(manifest, indent=2, sort_keys=True) + "\n"

    args = Namespace(
        results_root=results_root,
        ledger_root=ledger_root,
        manifest_path=manifest_path,
        json=False,
    )
    assert _handle_write_unfinished_manifest(args) == 0
    assert capsys.readouterr().out == (
        f"manifest_path={manifest_path}\nidentity_count=1\nactive_chat_models=model-a,model-b\n"
    )


def test_qwen_tail_handlers_forward_paths_and_preserve_json_contract(
    monkeypatch, tmp_path, capsys
) -> None:
    canonical_root = tmp_path / "canonical"
    tail_root = tmp_path / "tail"
    remote_root = "/workspace/results/tail"
    prepare_report = {
        "tail_results_root": str(tail_root),
        "manifest_path": str(tail_root / "shard_manifest.json"),
        "identity_count": 2,
        "config_path": str(tail_root / "runtime_config.yaml"),
    }
    preflight_report = {
        "tail_results_root": str(tail_root),
        "resume_preflight": {"can_resume": True, "pending_count": 3},
        "repo_root": str(tmp_path / "repo"),
        "canonical_results_root": str(canonical_root),
    }
    prepare_forwarded = []
    preflight_forwarded = []

    def fake_prepare(*, canonical_results_root, tail_results_root, remote_output_root):
        prepare_forwarded.append((canonical_results_root, tail_results_root, remote_output_root))
        return prepare_report

    def fake_preflight(
        *, repo_root, canonical_results_root, tail_results_root, watcher_status_path
    ):
        preflight_forwarded.append(
            (repo_root, canonical_results_root, tail_results_root, watcher_status_path)
        )
        return preflight_report

    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.prepare_qwen_algo1_tail_bundle",
        fake_prepare,
    )
    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.build_qwen_algo1_tail_preflight_report",
        fake_preflight,
    )

    assert (
        _handle_prepare_qwen_algo1_tail(
            Namespace(
                canonical_results_root=canonical_root,
                tail_results_root=tail_root,
                remote_output_root=remote_root,
                json=True,
            )
        )
        == 0
    )
    assert capsys.readouterr().out == json.dumps(prepare_report, indent=2, sort_keys=True) + "\n"

    watcher_path = tmp_path / "watcher.json"
    assert (
        _handle_qwen_algo1_tail_preflight(
            Namespace(
                repo_root=tmp_path / "repo",
                canonical_results_root=canonical_root,
                tail_results_root=tail_root,
                watcher_status_path=watcher_path,
                json=True,
            )
        )
        == 0
    )
    assert capsys.readouterr().out == json.dumps(preflight_report, indent=2, sort_keys=True) + "\n"
    assert prepare_forwarded == [(canonical_root, tail_root, remote_root)]
    assert preflight_forwarded == [(tmp_path / "repo", canonical_root, tail_root, watcher_path)]


def test_drain_remaining_report_forwards_plan_and_supervisor_arguments(
    monkeypatch, tmp_path
) -> None:
    common = {
        "repo_root": tmp_path / "repo",
        "results_root": tmp_path / "results",
        "ssh_command": "ssh example",
        "state_file": tmp_path / "state.json",
        "phase": "safe",
        "full_coverage": True,
        "root_name_contains": "olmo",
    }
    args = Namespace(
        **common,
        plan_only=True,
        poll_seconds=12.5,
        stale_after_seconds=99.0,
        quick_resume_script=tmp_path / "resume.sh",
    )
    plan_calls = []
    supervisor_calls = []

    def fake_plan(**kwargs):
        plan_calls.append(kwargs)
        return {"mode": "plan"}

    def fake_supervisor(**kwargs):
        supervisor_calls.append(kwargs)
        return {"mode": "supervisor"}

    monkeypatch.setattr("llm_conceptual_modeling.commands.run.build_drain_plan", fake_plan)
    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.run_drain_supervisor",
        fake_supervisor,
    )

    assert _drain_remaining_report(args) == {"mode": "plan"}
    assert plan_calls == [common]
    assert supervisor_calls == []

    args.plan_only = False
    assert _drain_remaining_report(args) == {"mode": "supervisor"}
    assert supervisor_calls == [
        {
            **common,
            "poll_seconds": 12.5,
            "stale_after_seconds": 99.0,
            "quick_resume_script": tmp_path / "resume.sh",
        }
    ]


def test_drain_remaining_handler_preserves_defaults_and_json_contract(
    monkeypatch, tmp_path, capsys
) -> None:
    state_file = tmp_path / "state.json"
    reports = [
        {"state_file": str(state_file)},
        {
            "state_file": str(state_file),
            "safe_queue_count": 3,
            "risky_queue_count": 2,
            "adopted_results_root": "/tmp/adopted",
        },
    ]
    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run._drain_remaining_report",
        lambda _args: reports.pop(0),
    )

    assert _handle_drain_remaining(Namespace(json=False)) == 0
    assert capsys.readouterr().out == (
        f"state_file={state_file}\nsafe_queue_count=0\nrisky_queue_count=0\n"
    )

    assert _handle_drain_remaining(Namespace(json=True)) == 0
    expected = {
        "state_file": str(state_file),
        "safe_queue_count": 3,
        "risky_queue_count": 2,
        "adopted_results_root": "/tmp/adopted",
    }
    assert capsys.readouterr().out == json.dumps(expected, indent=2, sort_keys=True) + "\n"

    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run._drain_remaining_report",
        lambda _args: expected,
    )
    assert _handle_drain_remaining(Namespace(json=False)) == 0
    assert capsys.readouterr().out == (
        f"state_file={state_file}\n"
        "safe_queue_count=3\n"
        "risky_queue_count=2\n"
        "adopted_results_root=/tmp/adopted\n"
    )


def test_drain_status_handler_forwards_path_and_preserves_missing_field_contract(
    monkeypatch, tmp_path, capsys
) -> None:
    state_file = tmp_path / "state.json"
    full_report = {
        "health": "healthy",
        "current_results_root": "/tmp/results",
        "current_phase": "safe",
    }
    calls = []

    def fake_read(path):
        calls.append(path)
        return full_report

    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.read_drain_state_report",
        fake_read,
    )
    assert _handle_drain_status(Namespace(state_file=state_file, json=True)) == 0
    assert capsys.readouterr().out == json.dumps(full_report, indent=2, sort_keys=True) + "\n"

    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.read_drain_state_report",
        lambda _path: {"health": "healthy", "current_phase": "safe"},
    )
    assert _handle_drain_status(Namespace(state_file=state_file, json=False)) == 0
    assert capsys.readouterr().out == (
        f"state_file={state_file}\n"
        "health=healthy\n"
        "current_phase=safe\n"
        "current_results_root=unknown\n"
    )
    assert calls == [state_file]


def test_smoke_handler_forwards_run_controls_and_preserves_json_contract(
    monkeypatch, tmp_path, capsys
) -> None:
    config_path = tmp_path / "config.yaml"
    config_marker = object()
    spec_marker = object()
    selected = {}
    executed = {}
    summary = {"status": "dry-run", "pair_name": "pair"}

    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.load_hf_run_config",
        lambda _path: config_marker,
    )

    def fake_select(**kwargs):
        selected.update(kwargs)
        return spec_marker

    def fake_run(**kwargs):
        executed.update(kwargs)
        return summary

    monkeypatch.setattr("llm_conceptual_modeling.commands.run.select_run_spec", fake_select)
    monkeypatch.setattr("llm_conceptual_modeling.commands.run.run_single_spec", fake_run)
    output_root = tmp_path / "smoke"
    args = Namespace(
        config=config_path,
        algorithm="algo1",
        model="model",
        graph_source="graph",
        pair_name="pair",
        condition_bits="00000",
        decoding="greedy",
        num_beams=2,
        penalty_alpha=0.2,
        top_k=4,
        replication=3,
        output_root=output_root,
        dry_run=True,
        resume=True,
    )

    assert _handle_smoke(args) == 0
    assert selected["config"] is config_marker
    assert selected["algorithm"] == "algo1"
    assert selected["model"] == "model"
    assert selected["graph_source"] == "graph"
    assert selected["pair_name"] == "pair"
    assert selected["condition_bits"] == "00000"
    assert selected["replication"] == 3
    assert selected["decoding"].algorithm == "greedy"
    assert executed == {
        "spec": spec_marker,
        "output_root": output_root,
        "dry_run": True,
        "resume": True,
    }
    assert capsys.readouterr().out == json.dumps(summary, indent=2, sort_keys=True) + "\n"


def test_experiment_handler_loads_config_and_forwards_batch_arguments(
    monkeypatch, tmp_path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_marker = object()
    loaded = []
    provider_calls = []
    batch_calls = []

    def fake_load(path):
        loaded.append(path)
        return config_marker

    def fake_require(args, config):
        provider_calls.append((args, config))

    def fake_run_paper_batch(**kwargs):
        batch_calls.append(kwargs)

    monkeypatch.setattr("llm_conceptual_modeling.commands.run.load_hf_run_config", fake_load)
    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run._require_hf_transformers_provider",
        fake_require,
    )
    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.run_paper_batch",
        fake_run_paper_batch,
    )

    args = Namespace(
        config=config_path,
        provider="hf-transformers",
        run_target="algo2",
        output_root=tmp_path / "output",
        model=["model"],
        embedding_model="embedding",
        replications=3,
        resume=True,
        dry_run=True,
    )
    assert _handle_experiment_run(args) == 0
    assert loaded == [config_path]
    assert provider_calls == [(args, config_marker)]
    assert batch_calls == [
        {
            "output_root": args.output_root,
            "models": ["model"],
            "embedding_model": "embedding",
            "replications": 3,
            "algorithms": ("algo2",),
            "config": config_marker,
            "resume": True,
            "dry_run": True,
        }
    ]

    args.embedding_model = ""
    assert _handle_experiment_run(args) == 0
    assert batch_calls[-1]["embedding_model"] == ""


def test_prefetch_runtime_for_config_forwards_model_configuration(monkeypatch) -> None:
    calls = []

    class FakeRuntimeFactory:
        def prefetch_models(self, *, chat_models, embedding_model):
            calls.append((chat_models, embedding_model))
            return {"prefetched": True}

    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.run.build_runtime_factory",
        lambda: FakeRuntimeFactory(),
    )
    config = Namespace(
        models=Namespace(
            chat_models=["model-a", "model-b"],
            embedding_model="embedding-model",
        )
    )

    assert prefetch_runtime_for_config(config=config) == {"prefetched": True}
    assert calls == [(["model-a", "model-b"], "embedding-model")]


def test_load_optional_run_config_distinguishes_missing_and_present_paths(
    monkeypatch, tmp_path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_marker = object()
    loaded = []

    def fake_load(path):
        loaded.append(path)
        return config_marker

    monkeypatch.setattr("llm_conceptual_modeling.commands.run.load_hf_run_config", fake_load)
    assert _load_optional_run_config(Namespace()) is None
    assert _load_optional_run_config(Namespace(config=config_path)) is config_marker
    assert loaded == [config_path]


def test_run_provider_and_algorithm_helpers_preserve_errors_and_targets() -> None:
    with pytest.raises(ValueError) as exc_info:
        _require_hf_transformers_provider(Namespace(provider="mistral"), None)
    assert str(exc_info.value) == (
        "The run command currently supports only --provider hf-transformers."
    )

    assert _run_algorithms("paper-batch") is None
    assert _run_algorithms("algo1") == ("algo1",)
    assert _run_algorithms("algo2") == ("algo2",)
    assert _run_algorithms("algo3") == ("algo3",)
    with pytest.raises(ValueError) as exc_info:
        _run_algorithms("unknown")
    assert str(exc_info.value) == "Unsupported run target: unknown"


def test_decoding_from_args_builds_all_supported_configs() -> None:
    greedy = _decoding_from_args(Namespace(decoding="greedy"))
    assert greedy.algorithm == "greedy"
    assert greedy.temperature == 0.0
    assert greedy.num_beams is None
    assert greedy.penalty_alpha is None
    assert greedy.top_k is None

    beam = _decoding_from_args(Namespace(decoding="beam", num_beams=6))
    assert beam.algorithm == "beam"
    assert beam.num_beams == 6
    assert beam.temperature == 0.0
    assert beam.penalty_alpha is None
    assert beam.top_k is None

    contrastive = _decoding_from_args(Namespace(decoding="contrastive", penalty_alpha=0.8, top_k=4))
    assert contrastive.algorithm == "contrastive"
    assert contrastive.penalty_alpha == 0.8
    assert contrastive.top_k == 4
    assert contrastive.temperature == 0.0


def _write_flat(path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
