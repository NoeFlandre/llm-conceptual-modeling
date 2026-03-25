from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis.variability_bundle import write_variability_bundle


def test_write_variability_bundle_creates_organized_bundle(tmp_path: Path) -> None:
    bundle_root = tmp_path / "output_variability"
    bundle_root.mkdir(parents=True)

    # Write pre-computed flat files that mimic the original audit output
    _write_lines(
        bundle_root / "algorithm_output_variability_summary.csv",
        [
            "algorithm,mean_pairwise_jaccard,exact_match_pair_rate,mean_edge_count,"
            "sample_std_edge_count,breadth_expansion_ratio",
            "algo1,0.998,0.998,14.0,0.005,1.004",
            "algo3,0.077,0.001,50.5,16.95,4.132",
        ],
    )
    _write_lines(
        bundle_root / "model_output_variability_summary.csv",
        [
            "model,algorithm,mean_pairwise_jaccard,exact_match_pair_rate,"
            "mean_edge_count,sample_std_edge_count,breadth_expansion_ratio",
            "gpt-4o,algo1,0.998,0.998,14.0,0.005,1.004",
            "gpt-4o,algo3,0.098,0.002,21.4,5.1,3.2",
        ],
    )
    _write_lines(
        bundle_root / "algo1_condition_output_variability.csv",
        [
            "model,source_input,mean_pairwise_jaccard,exact_match_pair_rate,"
            "mean_edge_count,sample_std_edge_count,breadth_expansion_ratio",
            "gpt-4o,data/results/algo1/gpt-4o/raw/file.csv,0.998,0.998,14.0,0.005,1.004",
        ],
    )
    _write_lines(
        bundle_root / "algo3_condition_output_variability.csv",
        [
            "model,source_input,mean_pairwise_jaccard,exact_match_pair_rate,"
            "mean_edge_count,sample_std_edge_count,breadth_expansion_ratio",
            "gpt-4o,data/results/algo3/gpt-4o/raw/file.csv,0.098,0.002,21.4,5.1,3.2",
        ],
    )
    _write_lines(
        bundle_root / "algo3_output_variability_by_depth.csv",
        [
            "Depth,mean_pairwise_jaccard,exact_match_pair_rate,"
            "mean_edge_count,breadth_expansion_ratio",
            "1,0.085,0.002,40.2,4.067",
            "2,0.069,0.000,60.8,4.197",
        ],
    )
    _write_lines(
        bundle_root / "algo3_output_variability_by_word_count.csv",
        [
            "Number of Words,mean_pairwise_jaccard,mean_edge_count",
            "3,0.080,36.1",
            "5,0.074,64.9",
        ],
    )
    _write_lines(
        bundle_root / "output_variability_extremes.csv",
        [
            "model,source_input,mean_pairwise_jaccard,exact_match_pair_rate,"
            "breadth_expansion_ratio,union_edge_count,mean_edge_count,algorithm,extreme_type",
            "gpt-4o,data/results/algo3/gpt-4o/raw/file.csv,0.0,0.0,5.0,200,40,algo3,lowest_jaccard",
        ],
    )

    write_variability_bundle(results_root=bundle_root, output_dir=bundle_root)

    # --- bundle-level artifacts ---
    assert (bundle_root / "README.md").exists()
    assert (bundle_root / "bundle_manifest.csv").exists()
    assert (bundle_root / "bundle_overview.csv").exists()

    # --- cross-algorithm summary preserved ---
    assert (bundle_root / "algorithm_output_variability_summary.csv").exists()
    assert (bundle_root / "model_output_variability_summary.csv").exists()
    assert (bundle_root / "output_variability_extremes.csv").exists()

    # --- per-algorithm subdirectories ---
    assert (bundle_root / "algo1" / "condition_output_variability.csv").exists()
    assert (bundle_root / "algo3" / "condition_output_variability.csv").exists()
    assert (bundle_root / "algo3" / "by_depth.csv").exists()
    assert (bundle_root / "algo3" / "by_word_count.csv").exists()

    # --- manifest structure ---
    manifest = pd.read_csv(bundle_root / "bundle_manifest.csv")
    assert len(manifest) > 0
    assert {"relative_path", "description"}.issubset(manifest.columns)
    assert all(manifest["description"].notna())

    # --- overview captures key numbers ---
    overview = pd.read_csv(bundle_root / "bundle_overview.csv")
    required = {
        "algorithm",
        "mean_pairwise_jaccard",
        "exact_match_pair_rate",
        "breadth_expansion_ratio",
    }
    assert required.issubset(overview.columns)

    # --- extremes preserved ---
    extremes = pd.read_csv(bundle_root / "output_variability_extremes.csv")
    assert set(extremes["algorithm"]) == {"algo3"}
    assert extremes["extreme_type"].iloc[0] == "lowest_jaccard"


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
