from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis.stability_bundle import write_stability_bundle


def test_write_stability_bundle_creates_organized_outputs(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir(parents=True)
    output_dir = tmp_path / "bundle"

    # Create flat stability files that mimic the original audit output structure
    _write_flat_file(
        results_root / "variability_incidence_by_algorithm.csv",
        [
            "algorithm,metric,condition_count,varying_condition_count,varying_condition_share",
            "algo1,accuracy,576,2,0.0035",
            "algo3,Recall,96,44,0.4583",
        ],
    )
    _write_flat_file(
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
    _write_flat_file(
        results_root / "algo1_condition_stability.csv",
        [
            (
                "source_input,Repetition,Explanation,Example,Counterexample,"
                "Array/List(1/-1),Tag/Adjacency(1/-1),metric,n,mean,sample_std,"
                "min,max,range_width,coefficient_of_variation"
            ),
            (
                "algo1/gpt-5/evaluated/metrics_sg1_sg2.csv,0,-1,-1,-1,1,1,"
                "accuracy,5,0.89,0.01,0.87,0.92,0.05,0.011"
            ),
        ],
    )
    _write_flat_file(
        results_root / "algo1_explanation_stability_by_level.csv",
        [
            (
                "Explanation,metric,condition_count,mean_cv,median_cv,"
                "mean_range_width,max_range_width"
            ),
            "-1,accuracy,288,0.00003,0.0,0.00007,0.042",
            "1,accuracy,288,0.0,0.0,0.0,0.0",
        ],
    )
    _write_flat_file(
        results_root / "algo1_explanation_variability_incidence.csv",
        [
            (
                "Explanation,metric,condition_count,varying_condition_count,"
                "varying_condition_share"
            ),
            "-1,accuracy,288,1,0.0035",
            "1,accuracy,288,0,0.0",
        ],
    )
    _write_flat_file(
        results_root / "algo2_condition_stability.csv",
        [
            (
                "source_input,Repetition,Convergence,Explanation,Example,"
                "Counterexample,Array/List(1/-1),Tag/Adjacency(1/-1),metric,"
                "n,mean,sample_std,min,max,range_width,coefficient_of_variation"
            ),
            (
                "algo2/gpt-5/evaluated/metrics_sg1_sg2.csv,0,-1,-1,-1,1,1,1,"
                "accuracy,5,0.86,0.02,0.83,0.89,0.06,0.023"
            ),
        ],
    )
    _write_flat_file(
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
    _write_flat_file(
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
    _write_flat_file(
        results_root / "algo3_condition_stability.csv",
        [
            (
                "source_input,Repetition,Depth,Example,Counter-Example,"
                "Number of Words,metric,n,mean,sample_std,min,max,"
                "range_width,coefficient_of_variation"
            ),
            (
                "algo3/gpt-5/evaluated/method3_results_evaluated_gpt5.csv,"
                "0,1,1,-1,3,Recall,5,0.03,0.01,0.01,0.05,0.04,0.33"
            ),
        ],
    )
    _write_flat_file(
        results_root / "algo3_depth_stability_by_level.csv",
        [
            (
                "Depth,metric,condition_count,mean_cv,median_cv,"
                "mean_range_width,max_range_width"
            ),
            "1,Recall,48,3.62,3.87,0.182,1.0",
            "2,Recall,48,2.95,2.89,0.491,1.0",
        ],
    )
    _write_flat_file(
        results_root / "algo3_depth_variability_incidence.csv",
        [
            (
                "Depth,metric,condition_count,varying_condition_count,"
                "varying_condition_share"
            ),
            "1,Recall,48,17,0.354",
            "2,Recall,48,27,0.562",
        ],
    )
    _write_flat_file(
        results_root / "algo3_number_of_words_stability_by_level.csv",
        [
            (
                "Number of Words,metric,condition_count,mean_cv,median_cv,"
                "mean_range_width,max_range_width"
            ),
            "3,Recall,48,3.5,3.5,0.18,0.9",
            "5,Recall,48,3.8,3.9,0.20,1.0",
        ],
    )

    write_stability_bundle(results_root=results_root, output_dir=output_dir)

    # --- bundle-level artifacts ---
    assert (output_dir / "README.md").exists()
    assert (output_dir / "bundle_manifest.csv").exists()
    assert (output_dir / "bundle_overview.csv").exists()
    assert (output_dir / "variability_incidence_by_algorithm.csv").exists()
    assert (output_dir / "overall_metric_stability_by_algorithm.csv").exists()

    # --- per-algorithm subdirectories ---
    assert (output_dir / "algo1" / "condition_stability.csv").exists()
    assert (output_dir / "algo1" / "explanation_stability_by_level.csv").exists()
    assert (output_dir / "algo1" / "explanation_variability_incidence.csv").exists()
    assert (output_dir / "algo2" / "condition_stability.csv").exists()
    assert (output_dir / "algo2" / "convergence_stability_by_level.csv").exists()
    assert (output_dir / "algo2" / "convergence_variability_incidence.csv").exists()
    assert (output_dir / "algo3" / "condition_stability.csv").exists()
    assert (output_dir / "algo3" / "depth_stability_by_level.csv").exists()
    assert (output_dir / "algo3" / "depth_variability_incidence.csv").exists()
    assert (output_dir / "algo3" / "number_of_words_stability_by_level.csv").exists()

    # --- manifest structure ---
    manifest = pd.read_csv(output_dir / "bundle_manifest.csv")
    assert len(manifest) > 0
    assert {"algorithm", "factor", "relative_path", "description"}.issubset(manifest.columns)
    # Every entry must have a non-empty description
    assert all(manifest["description"].notna())
    assert all(manifest["description"].str.len() > 0)

    # --- overview captures cross-algorithm metrics ---
    overview = pd.read_csv(output_dir / "bundle_overview.csv")
    assert set(overview["algorithm"]) == {"algo1", "algo3"}
    assert set(overview["metric"]) == {"accuracy", "Recall"}
    assert {"mean_cv", "median_cv", "condition_count"}.issubset(overview.columns)

    # --- variability incidence is reorganized correctly ---
    var_inc = pd.read_csv(output_dir / "variability_incidence_by_algorithm.csv")
    assert set(var_inc["algorithm"]) == {"algo1", "algo3"}
    assert "varying_condition_share" in var_inc.columns

    # --- convergence=1 in algo2 is perfectly stable (zero varying conditions) ---
    conv_inc = pd.read_csv(output_dir / "algo2" / "convergence_variability_incidence.csv")
    conv_1 = conv_inc[conv_inc["Convergence"] == 1].iloc[0]
    assert conv_1["varying_condition_count"] == 0
    assert conv_1["varying_condition_share"] == 0.0


def _write_flat_file(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
