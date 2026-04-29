from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from llm_conceptual_modeling.algo3.evaluation import compute_recall_for_row, parse_edge_list
from llm_conceptual_modeling.analysis._baseline_compare import (
    _baseline_repetitions,
    _stable_random_seed,
)
from llm_conceptual_modeling.analysis.baseline_bundle import (
    _sample_baseline_edges,
    _scored_connection_count,
    write_baseline_comparison_bundle,
)
from llm_conceptual_modeling.common.connection_eval import find_valid_connections
from llm_conceptual_modeling.common.literals import parse_python_literal


class TestWriteBaselineComparisonBundle:
    def test_pre_generated_bundle_has_manifest_and_readme(self, fixture_bundle_path: Path) -> None:
        bundle_path = fixture_bundle_path
        assert (bundle_path / "bundle_manifest.csv").exists()
        assert (bundle_path / "README.md").exists()

    def test_manifest_contains_required_columns(self, fixture_bundle_path: Path) -> None:
        bundle_path = fixture_bundle_path
        manifest = pd.read_csv(bundle_path / "bundle_manifest.csv")
        assert "file" in manifest.columns
        assert "description" in manifest.columns

    def test_advantage_summary_has_expected_structure(self, fixture_bundle_path: Path) -> None:
        bundle_path = fixture_bundle_path
        summary = pd.read_csv(bundle_path / "baseline_advantage_summary.csv")
        assert "algorithm" in summary.columns
        assert "baseline_strategy" in summary.columns
        assert "metric" in summary.columns
        assert "models_beating_baseline" in summary.columns
        assert len(summary) > 0

    def test_random_k_precision_summary_is_well_formed(self, fixture_bundle_path: Path) -> None:
        """Random-k precision rows should be present and reported with bounded counts.

        The fixture corpus is intentionally tiny, so the empirical winner can
        differ by algorithm after the baseline is scored through the same
        connection-evaluation pipeline as the LLM outputs.
        """
        bundle_path = fixture_bundle_path
        summary = pd.read_csv(bundle_path / "baseline_advantage_summary.csv")
        precision_rows = summary[
            (summary["metric"] == "precision") & (summary["baseline_strategy"] == "random-k")
        ]
        assert len(precision_rows) > 0
        for _, row in precision_rows.iterrows():
            if row["algorithm"] in ("algo1", "algo2"):
                assert 0 <= row["models_beating_baseline"] <= row["model_count"], (
                    f"{row['algorithm']} precision: beating count out of bounds: "
                    f"{row['models_beating_baseline']} of {row['model_count']}"
                )
                assert row["model_count"] > 0

    def test_algo3_random_baseline_summary_is_auditable(
        self,
        fixture_bundle_path: Path,
    ) -> None:
        bundle_path = fixture_bundle_path
        summary = pd.read_csv(bundle_path / "baseline_advantage_summary.csv")
        algo3_rows = summary[
            (summary["algorithm"] == "algo3") & (summary["baseline_strategy"] == "random-k")
        ]
        for _, row in algo3_rows.iterrows():
            assert 0 <= row["models_beating_baseline"] <= row["model_count"]
            assert row["model_count"] > 0

    def test_random_k_baseline_produces_nontrivial_comparison(
        self,
        fixture_bundle_path: Path,
    ) -> None:
        """The random-k baseline should produce meaningful comparison output:
        per-model comparison files should exist with non-empty data."""
        bundle_path = fixture_bundle_path
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
        } == set(summary["baseline_strategy"].unique())

    def test_random_k_samples_admissible_cross_pairs_not_mother_edges(self) -> None:
        mother_edges = [("a", "b"), ("x", "y")]
        subgraph1_edges = [("a", "b")]
        subgraph2_edges = [("x", "y")]

        sampled = _sample_baseline_edges(
            baseline_strategy="random-k",
            k=4,
            mother_edges=mother_edges,
            subgraph1_edges=subgraph1_edges,
            subgraph2_edges=subgraph2_edges,
            random_seed=17,
        )

        admissible_pairs = {
            ("a", "x"),
            ("a", "y"),
            ("b", "x"),
            ("b", "y"),
        }
        assert sampled == admissible_pairs
        assert sampled.isdisjoint(set(mother_edges))

    def test_scored_k_counts_cross_connections_not_raw_edges(self) -> None:
        subgraph1_edges = [("a", "b")]
        subgraph2_edges = [("x", "y")]
        raw_edges = [("b", "bridge"), ("bridge", "x"), ("bridge", "unused")]

        assert len(raw_edges) == 3
        assert (
            _scored_connection_count(
                raw_edges,
                subgraph1_edges=subgraph1_edges,
                subgraph2_edges=subgraph2_edges,
            )
            == 4
        )

    def test_random_k_grouped_output_uses_five_repetitions(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "bundle"
        results_root = _build_fixture_results_root(tmp_path)

        write_baseline_comparison_bundle(
            results_root=str(results_root),
            output_dir=str(output_dir),
            random_repetitions=5,
        )

        row_level = pd.read_csv(output_dir / "row_level_baseline_comparison.csv")
        random_rows = row_level[row_level["baseline_strategy"] == "random-k"]
        assert set(random_rows["baseline_repetition"].dropna().astype(int)) == {0, 1, 2, 3, 4}

        grouped = pd.read_csv(output_dir / "all_models_vs_baseline.csv")
        random_grouped = grouped[grouped["baseline_strategy"] == "random-k"]
        assert {"baseline_ci95_low", "baseline_ci95_high", "mean_k", "row_count"}.issubset(
            random_grouped.columns
        )

    def test_per_model_summary_is_emitted_with_random_interval_columns(
        self,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "bundle"
        results_root = _build_fixture_results_root(tmp_path)

        write_baseline_comparison_bundle(
            results_root=str(results_root),
            output_dir=str(output_dir),
            random_repetitions=5,
        )

        summary = pd.read_csv(output_dir / "per_model_baseline_summary.csv")
        assert {
            "algorithm",
            "model",
            "metric",
            "llm_mean",
            "random_k_mean",
            "random_k_ci95_low",
            "random_k_ci95_high",
            "wordnet_ontology_match_mean",
            "mean_k",
            "row_count",
        }.issubset(summary.columns)

    def test_algo3_baseline_outputs_are_recall_only(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "bundle"
        results_root = _build_fixture_results_root(tmp_path)

        write_baseline_comparison_bundle(
            results_root=str(results_root),
            output_dir=str(output_dir),
            random_repetitions=5,
        )

        row_level = pd.read_csv(output_dir / "row_level_baseline_comparison.csv")
        grouped = pd.read_csv(output_dir / "algo3_model_vs_baseline.csv")
        summary = pd.read_csv(output_dir / "per_model_baseline_summary.csv")

        assert set(row_level[row_level["algorithm"] == "algo3"]["metric"]) == {"recall"}
        assert set(grouped["metric"]) == {"recall"}
        assert set(summary[summary["algorithm"] == "algo3"]["metric"]) == {"recall"}

    def test_all_models_vs_baseline_overwrites_not_appends(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "bundle"
        results_root = _build_fixture_results_root(tmp_path)
        stale_tex = output_dir / "deepseek_gpt_gemini_wordnet_randomk.tex"
        output_dir.mkdir(parents=True)
        stale_tex.write_text("stale", encoding="utf-8")

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
        assert not stale_tex.exists()

    def test_llm_means_are_invariant_across_baseline_strategies(
        self,
        fixture_bundle_path: Path,
    ) -> None:
        bundle_path = fixture_bundle_path
        comparison = pd.read_csv(bundle_path / "all_models_vs_baseline.csv")

        for _, metric_rows in comparison.groupby(["algorithm", "model", "metric"], dropna=False):
            llm_means = metric_rows["llm_mean"].tolist()
            assert llm_means, "Expected non-empty comparison rows"
            reference = llm_means[0]
            for llm_mean in llm_means[1:]:
                assert math.isclose(reference, llm_mean, rel_tol=0.0, abs_tol=1e-12), (
                    "LLM mean changed across baseline strategies for "
                    f"{metric_rows.iloc[0]['algorithm']} / {metric_rows.iloc[0]['model']} / "
                    f"{metric_rows.iloc[0]['metric']}: {llm_means}"
                )

    def test_llm_means_match_evaluated_metrics(
        self,
        fixture_results_root: Path,
        fixture_bundle_path: Path,
    ) -> None:
        results_root = fixture_results_root
        bundle_path = fixture_bundle_path
        comparison = pd.read_csv(bundle_path / "all_models_vs_baseline.csv")

        expected_rows = []
        for eval_file in results_root.glob("algo1/*/evaluated/*.csv"):
            model = eval_file.parts[-3]
            evaluated = pd.read_csv(eval_file)
            for metric in ["accuracy", "precision", "recall"]:
                expected_rows.append(
                    {
                        "algorithm": "algo1",
                        "model": model,
                        "metric": metric,
                        "expected": float(evaluated[metric].mean()),
                    }
                )

        for eval_file in results_root.glob("algo2/*/evaluated/*.csv"):
            model = eval_file.parts[-3]
            evaluated = pd.read_csv(eval_file)
            for metric in ["accuracy", "precision", "recall"]:
                expected_rows.append(
                    {
                        "algorithm": "algo2",
                        "model": model,
                        "metric": metric,
                        "expected": float(evaluated[metric].mean()),
                    }
                )

        for eval_file in results_root.glob("algo3/*/evaluated/*.csv"):
            model = eval_file.parts[-3]
            evaluated = pd.read_csv(eval_file)
            expected_rows.append(
                {
                    "algorithm": "algo3",
                    "model": model,
                    "metric": "recall",
                    "expected": float(evaluated["Recall"].mean()),
                }
            )

        expected = pd.DataFrame.from_records(expected_rows)
        observed = (
            comparison.groupby(["algorithm", "model", "metric"], dropna=False)["llm_mean"]
            .first()
            .reset_index()
        )
        merged = observed.merge(expected, on=["algorithm", "model", "metric"], how="inner")

        for _, row in merged.iterrows():
            assert math.isclose(row["llm_mean"], row["expected"], rel_tol=0.0, abs_tol=1e-12), (
                "Bundle LLM mean does not match evaluated metric for "
                f"{row['algorithm']} / {row['model']} / {row['metric']}: "
                f"observed={row['llm_mean']}, expected={row['expected']}"
            )

    def test_baseline_means_match_algo12_evaluation_pipeline(
        self,
        fixture_results_root: Path,
        fixture_bundle_path: Path,
    ) -> None:
        results_root = fixture_results_root
        bundle_path = fixture_bundle_path
        comparison = pd.read_csv(bundle_path / "all_models_vs_baseline.csv")

        expected_rows: list[dict[str, object]] = []
        for algorithm in ["algo1", "algo2"]:
            for raw_file in results_root.glob(f"{algorithm}/*/raw/*.csv"):
                model = raw_file.parts[-3]
                raw = pd.read_csv(raw_file)
                for baseline_strategy in [
                    "random-k",
                    "wordnet-ontology-match",
                ]:
                    metric_rows = []
                    for row_index, row in raw.iterrows():
                        llm_result_edges = _normalize_edges(parse_python_literal(row["Result"]))
                        mother_edges = _normalize_edges(parse_python_literal(row["graph"]))
                        subgraph1_edges = _normalize_edges(parse_python_literal(row["subgraph1"]))
                        subgraph2_edges = _normalize_edges(parse_python_literal(row["subgraph2"]))
                        k = _scored_connection_count(
                            llm_result_edges,
                            subgraph1_edges=subgraph1_edges,
                            subgraph2_edges=subgraph2_edges,
                        )

                        ground_truth = find_valid_connections(
                            mother_edges,
                            subgraph1_edges,
                            subgraph2_edges,
                        )
                        for baseline_repetition in _baseline_repetitions(
                            baseline_strategy,
                            5,
                        ):
                            baseline_edges = _sample_baseline_edges(
                                baseline_strategy=baseline_strategy,
                                k=k,
                                mother_edges=mother_edges,
                                subgraph1_edges=subgraph1_edges,
                                subgraph2_edges=subgraph2_edges,
                                random_seed=_stable_random_seed(
                                    algorithm,
                                    model,
                                    raw_file.name.replace("algorithm1_results_", "metrics_")
                                    if algorithm == "algo1"
                                    else raw_file.name.replace("algorithm2_results_", "metrics_"),
                                    row_index,
                                    baseline_strategy,
                                    baseline_repetition,
                                ),
                            )

                            proposed_edges = [
                                *subgraph1_edges,
                                *subgraph2_edges,
                                *sorted(baseline_edges),
                            ]
                            generated_connections = find_valid_connections(
                                proposed_edges,
                                subgraph1_edges,
                                subgraph2_edges,
                            )
                            nodes1 = {node for edge in subgraph1_edges for node in edge}
                            nodes2 = {node for edge in subgraph2_edges for node in edge}
                            tp = len(generated_connections & ground_truth)
                            fp = len(generated_connections - ground_truth)
                            fn = len(ground_truth - generated_connections)
                            tn = (len(nodes1) * len(nodes2)) - (tp + fp + fn)
                            total = tp + fp + fn + tn
                            metric_rows.append(
                                {
                                    "accuracy": (tp + tn) / total if total > 0 else 0.0,
                                    "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
                                    "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                                }
                            )

                    metric_frame = pd.DataFrame.from_records(metric_rows)
                    for metric in ["accuracy", "precision", "recall"]:
                        expected_rows.append(
                            {
                                "algorithm": algorithm,
                                "model": model,
                                "baseline_strategy": baseline_strategy,
                                "metric": metric,
                                "expected": float(metric_frame[metric].mean()),
                            }
                        )

        expected = pd.DataFrame.from_records(expected_rows)
        observed = comparison[
            comparison["algorithm"].isin(["algo1", "algo2"])
        ][
            ["algorithm", "model", "baseline_strategy", "metric", "baseline_mean"]
        ]
        merged = observed.merge(
            expected,
            on=["algorithm", "model", "baseline_strategy", "metric"],
            how="inner",
        )

        for _, row in merged.iterrows():
            assert math.isclose(
                row["baseline_mean"],
                row["expected"],
                rel_tol=0.0,
                abs_tol=1e-12,
            ), (
                "Bundle baseline mean does not match the ALGO1/2 evaluation pipeline for "
                f"{row['algorithm']} / {row['model']} / {row['baseline_strategy']} / "
                f"{row['metric']}: observed={row['baseline_mean']}, "
                f"expected={row['expected']}"
            )

    def test_algo3_recall_baseline_matches_evaluation_pipeline(
        self,
        fixture_results_root: Path,
        fixture_bundle_path: Path,
    ) -> None:
        results_root = fixture_results_root
        bundle_path = fixture_bundle_path
        comparison = pd.read_csv(bundle_path / "all_models_vs_baseline.csv")

        expected_rows: list[dict[str, object]] = []
        for eval_file in results_root.glob("algo3/*/evaluated/*.csv"):
            model = eval_file.parts[-3]
            evaluated = pd.read_csv(eval_file)
            for baseline_strategy in [
                "random-k",
                "wordnet-ontology-match",
            ]:
                recalls = []
                for row_index, row in evaluated.iterrows():
                    source_edges = _normalize_edges(row.get("Source Graph", []))
                    target_edges = _normalize_edges(row.get("Target Graph", []))
                    mother_edges = _normalize_edges(row.get("Mother Graph", []))
                    llm_edges = _normalize_edges(row.get("Results", []))
                    k = _scored_connection_count(
                        llm_edges,
                        subgraph1_edges=source_edges,
                        subgraph2_edges=target_edges,
                    )
                    for baseline_repetition in _baseline_repetitions(
                        baseline_strategy,
                        5,
                    ):
                        baseline_edges = _sample_baseline_edges(
                            baseline_strategy=baseline_strategy,
                            k=k,
                            mother_edges=mother_edges,
                            subgraph1_edges=source_edges,
                            subgraph2_edges=target_edges,
                            random_seed=_stable_random_seed(
                                "algo3",
                                model,
                                eval_file.name,
                                row_index,
                                baseline_strategy,
                                baseline_repetition,
                            ),
                        )
                        recalls.append(
                            compute_recall_for_row(
                                source_edges,
                                target_edges,
                                mother_edges,
                                sorted(baseline_edges),
                            )
                        )

                expected_rows.append(
                    {
                        "algorithm": "algo3",
                        "model": model,
                        "baseline_strategy": baseline_strategy,
                        "metric": "recall",
                        "expected": float(sum(recalls) / len(recalls)),
                    }
                )

        expected = pd.DataFrame.from_records(expected_rows)
        observed = comparison[
            (comparison["algorithm"] == "algo3") & (comparison["metric"] == "recall")
        ][
            ["algorithm", "model", "baseline_strategy", "metric", "baseline_mean"]
        ]
        merged = observed.merge(
            expected,
            on=["algorithm", "model", "baseline_strategy", "metric"],
            how="inner",
        )

        for _, row in merged.iterrows():
            assert math.isclose(
                row["baseline_mean"],
                row["expected"],
                rel_tol=0.0,
                abs_tol=1e-12,
            ), (
                "Bundle ALGO3 recall baseline does not match compute_recall_for_row for "
                f"{row['model']} / {row['baseline_strategy']}: "
                f"observed={row['baseline_mean']}, expected={row['expected']}"
            )


def _copy_fixture(source_path: str, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(Path(source_path).read_text())


@pytest.fixture(scope="module")
def fixture_results_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("baseline-bundle-results")
    return _build_fixture_results_root(root)


@pytest.fixture(scope="module")
def fixture_bundle_path(
    tmp_path_factory: pytest.TempPathFactory,
    fixture_results_root: Path,
) -> Path:
    output_dir = tmp_path_factory.mktemp("baseline-bundle-output")
    write_baseline_comparison_bundle(
        results_root=str(fixture_results_root),
        output_dir=str(output_dir),
    )
    return output_dir


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
    return _populate_fixture_results_root(results_root)


def _populate_fixture_results_root(results_root: Path) -> Path:
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


def _normalize_edges(value: object) -> list[tuple[str, str]]:
    if isinstance(value, str):
        try:
            parsed = parse_python_literal(value)
        except Exception:
            return parse_edge_list(value)
    else:
        parsed = value
    edges: list[tuple[str, str]] = []
    if isinstance(parsed, (list, tuple, set)):
        for item in parsed:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                edges.append((str(item[0]).strip(), str(item[1]).strip()))
    return edges
