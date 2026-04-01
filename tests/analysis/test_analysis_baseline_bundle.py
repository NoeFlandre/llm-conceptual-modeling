from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis.baseline_bundle import write_baseline_comparison_bundle


class TestWriteBaselineComparisonBundle:
    def test_pre_generated_bundle_has_manifest_and_readme(self, tmp_path: Path) -> None:
        bundle_path = _build_fixture_bundle(tmp_path)
        assert (bundle_path / "bundle_manifest.csv").exists()
        assert (bundle_path / "README.md").exists()

    def test_manifest_contains_required_columns(self, tmp_path: Path) -> None:
        bundle_path = _build_fixture_bundle(tmp_path)
        manifest = pd.read_csv(bundle_path / "bundle_manifest.csv")
        assert "file" in manifest.columns
        assert "description" in manifest.columns

    def test_advantage_summary_has_expected_structure(self, tmp_path: Path) -> None:
        bundle_path = _build_fixture_bundle(tmp_path)
        summary = pd.read_csv(bundle_path / "baseline_advantage_summary.csv")
        assert "algorithm" in summary.columns
        assert "baseline_strategy" in summary.columns
        assert "metric" in summary.columns
        assert "models_beating_baseline" in summary.columns
        assert len(summary) > 0

    def test_llm_beats_random_k_baseline_on_precision(self, tmp_path: Path) -> None:
        """The random-k baseline samples k edges from the mother graph (no subgraph
        structure). ALGO1/2 LLM precision (~25-40%) substantially exceeds the
        baseline's expected precision (~4-5%), so all ALGO1/2 models should
        beat the baseline on precision."""
        bundle_path = _build_fixture_bundle(tmp_path)
        summary = pd.read_csv(bundle_path / "baseline_advantage_summary.csv")
        precision_rows = summary[
            (summary["metric"] == "precision") & (summary["baseline_strategy"] == "random-k")
        ]
        for _, row in precision_rows.iterrows():
            if row["algorithm"] in ("algo1", "algo2"):
                assert row["models_beating_baseline"] == row["model_count"], (
                    f"{row['algorithm']} precision: expected all {row['model_count']} "
                    f"models to beat baseline, got {row['models_beating_baseline']}"
                )
                assert row["best_model_delta"] > 0, (
                    f"{row['algorithm']} precision: expected positive delta, "
                    f"got {row['best_model_delta']}"
                )

    def test_algo3_llm_loses_to_random_baseline(self, tmp_path: Path) -> None:
        """ALGO3 outputs are noisy and inconsistent; even random guessing
        (concentrated in a few correct edges) outperforms the LLM on all metrics.
        This confirms that ALGO3's approach does not add value over random."""
        bundle_path = _build_fixture_bundle(tmp_path)
        summary = pd.read_csv(bundle_path / "baseline_advantage_summary.csv")
        algo3_rows = summary[
            (summary["algorithm"] == "algo3") & (summary["baseline_strategy"] == "random-k")
        ]
        for _, row in algo3_rows.iterrows():
            assert row["models_beating_baseline"] == 0, (
                f"{row['algorithm']} {row['metric']}: expected 0 models "
                f"beating baseline, got {row['models_beating_baseline']}"
            )
            assert row["best_model_delta"] < 0, (
                f"{row['algorithm']} {row['metric']}: expected negative delta, "
                f"got {row['best_model_delta']}"
            )

    def test_random_k_baseline_produces_nontrivial_comparison(self, tmp_path: Path) -> None:
        """The random-k baseline should produce meaningful comparison output:
        per-model comparison files should exist with non-empty data."""
        bundle_path = _build_fixture_bundle(tmp_path)
        for algo in ["algo1", "algo2", "algo3"]:
            comp_file = bundle_path / f"{algo}_model_vs_baseline.csv"
            assert comp_file.exists(), f"{algo} comparison file missing"
            df = pd.read_csv(comp_file)
            assert len(df) > 0, f"{algo} comparison has no rows"
            assert "baseline_strategy" in df.columns
            assert "llm_mean" in df.columns
            assert "baseline_mean" in df.columns
            assert "mean_delta" in df.columns

    def test_strategy_specific_summary_tracks_new_reviewer_baselines(self, tmp_path: Path) -> None:
        results_root = tmp_path / "results"
        output_dir = tmp_path / "bundle"
        _copy_fixture(
            "tests/reference_fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg1_sg2.csv",
            results_root / "algo1" / "gpt-5" / "evaluated" / "metrics_sg1_sg2.csv",
        )
        _copy_fixture(
            "tests/reference_fixtures/legacy/algo1/gpt-5/raw/algorithm1_results_sg1_sg2.csv",
            results_root / "algo1" / "gpt-5" / "raw" / "algorithm1_results_sg1_sg2.csv",
        )

        write_baseline_comparison_bundle(
            results_root=str(results_root),
            output_dir=str(output_dir),
        )

        summary = pd.read_csv(output_dir / "baseline_advantage_summary.csv")

        assert "baseline_strategy" in summary.columns
        assert {
            "random-k",
            "wordnet-ontology-match",
            "edit-distance",
        }.issubset(set(summary["baseline_strategy"].unique()))

    def test_all_models_vs_baseline_overwrites_not_appends(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "bundle"
        results_root = _build_fixture_results_root(tmp_path)

        write_baseline_comparison_bundle(
            results_root=str(results_root),
            output_dir=str(output_dir),
        )
        write_baseline_comparison_bundle(
            results_root=str(results_root),
            output_dir=str(output_dir),
        )

        all_df = pd.read_csv(output_dir / "all_models_vs_baseline.csv")
        unique = all_df.groupby(["algorithm", "model", "baseline_strategy", "metric"]).ngroups
        total = len(all_df)

        assert total == unique, (
            f"all_models_vs_baseline has {total} rows but only {unique} unique "
            f"(algorithm, model, baseline_strategy, metric) combos."
        )


def _copy_fixture(source_path: str, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(Path(source_path).read_text())


def _build_fixture_bundle(tmp_path: Path) -> Path:
    results_root = _build_fixture_results_root(tmp_path)
    output_dir = tmp_path / "bundle"

    write_baseline_comparison_bundle(
        results_root=str(results_root),
        output_dir=str(output_dir),
    )
    return output_dir


def _build_fixture_results_root(tmp_path: Path) -> Path:
    results_root = tmp_path / "results"
    _copy_fixture(
        "tests/reference_fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg1_sg2.csv",
        results_root / "algo1" / "gpt-5" / "evaluated" / "metrics_sg1_sg2.csv",
    )
    _copy_fixture(
        "tests/reference_fixtures/legacy/algo1/gpt-5/raw/algorithm1_results_sg1_sg2.csv",
        results_root / "algo1" / "gpt-5" / "raw" / "algorithm1_results_sg1_sg2.csv",
    )
    _copy_fixture(
        "tests/reference_fixtures/legacy/algo2/gpt-5/evaluated/metrics_sg1_sg2.csv",
        results_root / "algo2" / "gpt-5" / "evaluated" / "metrics_sg1_sg2.csv",
    )
    _copy_fixture(
        "tests/reference_fixtures/legacy/algo2/gpt-5/raw/algorithm2_results_sg1_sg2.csv",
        results_root / "algo2" / "gpt-5" / "raw" / "algorithm2_results_sg1_sg2.csv",
    )
    _copy_fixture(
        "tests/reference_fixtures/legacy/algo3/gpt-5/evaluated/method3_results_evaluated_gpt5.csv",
        results_root / "algo3" / "gpt-5" / "evaluated" / "method3_results_evaluated_gpt5.csv",
    )
    return results_root
