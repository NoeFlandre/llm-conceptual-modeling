from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis.baseline_bundle import write_baseline_comparison_bundle

# tests/analysis/test_xxx.py -> parents[0]=tests/analysis, [1]=tests, [2]=project root
_REAL_RESULTS_ROOT = Path(__file__).parents[2] / "data" / "results"
_BUNDLE_PATH = (
    Path(__file__).parents[2]
    / "data"
    / "analysis_artifacts"
    / "revision_tracker"
    / "baseline_comparison"
)


class TestWriteBaselineComparisonBundle:
    def test_pre_generated_bundle_has_manifest_and_readme(self) -> None:
        """The pre-generated bundle in revision_tracker has the expected structure."""
        assert (_BUNDLE_PATH / "bundle_manifest.csv").exists()
        assert (_BUNDLE_PATH / "README.md").exists()

    def test_manifest_contains_required_columns(self) -> None:
        manifest = pd.read_csv(_BUNDLE_PATH / "bundle_manifest.csv")
        assert "file" in manifest.columns
        assert "description" in manifest.columns

    def test_advantage_summary_has_expected_structure(self) -> None:
        summary = pd.read_csv(_BUNDLE_PATH / "baseline_advantage_summary.csv")
        assert "algorithm" in summary.columns
        assert "metric" in summary.columns
        assert "models_beating_baseline" in summary.columns
        assert len(summary) > 0

    def test_llm_beats_random_k_baseline_on_precision(self) -> None:
        """The random-k baseline samples k edges from the mother graph (no subgraph
        structure). ALGO1/2 LLM precision (~25-40%) substantially exceeds the
        baseline's expected precision (~4-5%), so all ALGO1/2 models should
        beat the baseline on precision."""
        summary = pd.read_csv(_BUNDLE_PATH / "baseline_advantage_summary.csv")
        precision_rows = summary[summary["metric"] == "precision"]
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

    def test_algo3_llm_loses_to_random_baseline(self) -> None:
        """ALGO3 outputs are noisy and inconsistent; even random guessing
        (concentrated in a few correct edges) outperforms the LLM on all metrics.
        This confirms that ALGO3's approach does not add value over random."""
        summary = pd.read_csv(_BUNDLE_PATH / "baseline_advantage_summary.csv")
        algo3_rows = summary[summary["algorithm"] == "algo3"]
        for _, row in algo3_rows.iterrows():
            assert row["models_beating_baseline"] == 0, (
                f"{row['algorithm']} {row['metric']}: expected 0 models "
                f"beating baseline, got {row['models_beating_baseline']}"
            )
            assert row["best_model_delta"] < 0, (
                f"{row['algorithm']} {row['metric']}: expected negative delta, "
                f"got {row['best_model_delta']}"
            )

    def test_random_k_baseline_produces_nontrivial_comparison(self) -> None:
        """The random-k baseline should produce meaningful comparison output:
        per-model comparison files should exist with non-empty data."""
        for algo in ["algo1", "algo2", "algo3"]:
            comp_file = _BUNDLE_PATH / f"{algo}_model_vs_baseline.csv"
            assert comp_file.exists(), f"{algo} comparison file missing"
            df = pd.read_csv(comp_file)
            assert len(df) > 0, f"{algo} comparison has no rows"
            assert "llm_mean" in df.columns
            assert "baseline_mean" in df.columns
            assert "mean_delta" in df.columns

    def test_all_models_vs_baseline_overwrites_not_appends(self, tmp_path: Path) -> None:
        """all_models_vs_baseline.csv must be overwritten (not appended to) on each run.

        Bug: the code used `if (all_out.exists() and False)` in the write logic,
        making the intended-overwrite branch unreachable and always appending instead.
        This test runs the generator twice on the same output dir and verifies the
        file still has exactly one row per (algorithm, model, metric)."""
        output_dir = tmp_path / "bundle"

        # First run: creates all_models_vs_baseline.csv
        write_baseline_comparison_bundle(
            results_root=str(_REAL_RESULTS_ROOT),
            output_dir=str(output_dir),
        )

        # Second run: must overwrite, not append
        write_baseline_comparison_bundle(
            results_root=str(_REAL_RESULTS_ROOT),
            output_dir=str(output_dir),
        )

        all_df = pd.read_csv(output_dir / "all_models_vs_baseline.csv")
        unique = all_df.groupby(["algorithm", "model", "metric"]).ngroups
        total = len(all_df)

        assert total == unique, (
            f"all_models_vs_baseline has {total} rows but only {unique} unique "
            f"(algorithm, model, metric) combos — rows are being duplicated on "
            "regeneration. The file should be overwritten, not appended to."
        )
