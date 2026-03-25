from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis.figures_bundle import write_figures_bundle


def test_figures_bundle_distributional_summary_aggregates_all_files_per_model(tmp_path: Path) -> None:
    """When a model has multiple evaluated source files, the distributional summary
    must be computed over ALL files for that model (not just the last one).

    Bug: the current implementation computes the summary inside the per-file loop,
    overwriting the same output file each time. The last file's data survives, but
    all previous files are lost. This test asserts the summary covers all files."""
    results_root = tmp_path / "results"
    results_root.mkdir(parents=True)
    output_dir = tmp_path / "bundle"

    # Two files for gpt-5 algo1 with DIFFERENT accuracy means
    # File A: accuracy mean = 0.90, File B: accuracy mean = 0.50
    # When aggregated (n=10), mean should be 0.70
    # When only file B survives (bug), mean should be 0.50, n=5
    _write_evaluated(
        results_root / "algo1" / "gpt-5" / "evaluated" / "metrics_sg1_sg2.csv",
        repetitions=[0, 1, 2, 3, 4],
        Explanation=1,
        Example=1,
        Counterexample=1,
        **{"Array/List(1/-1)": 1},
        **{"Tag/Adjacency(1/-1)": 1},
        accuracy=[0.90, 0.91, 0.90, 0.92, 0.91],  # mean ≈ 0.908
        precision=[0.35, 0.37, 0.36, 0.38, 0.35],
        recall=[0.08, 0.09, 0.08, 0.10, 0.09],
    )
    _write_evaluated(
        results_root / "algo1" / "gpt-5" / "evaluated" / "metrics_sg3_sg1.csv",
        repetitions=[0, 1, 2, 3, 4],
        Explanation=-1,
        Example=-1,
        Counterexample=-1,
        **{"Array/List(1/-1)": 1},
        **{"Tag/Adjacency(1/-1)": 1},
        accuracy=[0.50, 0.51, 0.50, 0.52, 0.51],  # mean ≈ 0.508
        precision=[0.20, 0.21, 0.20, 0.22, 0.20],
        recall=[0.05, 0.06, 0.05, 0.07, 0.05],
    )

    write_figures_bundle(results_root=results_root, output_dir=output_dir)

    dist = pd.read_csv(output_dir / "algo1" / "gpt-5" / "distributional_summary.csv")
    acc_row = dist[dist["metric"] == "accuracy"].iloc[0]

    # n must be 10 (5 rows from sg1_sg2 + 5 rows from sg3_sg1), not 5
    assert int(acc_row["n"]) == 10, (
        f"Expected n=10 (both files), got n={acc_row['n']}. "
        "Distributional summary should aggregate all files per model."
    )

    # Mean accuracy must be the combined mean, not just the last file's mean
    # Combined mean = (0.908 + 0.508) / 2 ≈ 0.708
    expected_mean = 0.70
    actual_mean = float(acc_row["mean"])
    assert abs(actual_mean - expected_mean) < 0.02, (
        f"Expected mean≈{expected_mean}, got {actual_mean:.4f}. "
        "Distributional summary should cover all files for the model."
    )


def test_figures_bundle_overview_covers_all_files_per_model(tmp_path: Path) -> None:
    """The bundle_overview must have one row per model-metric, not one per source file."""
    results_root = tmp_path / "results"
    results_root.mkdir(parents=True)
    output_dir = tmp_path / "bundle"

    _write_evaluated(
        results_root / "algo1" / "gpt-5" / "evaluated" / "metrics_sg1_sg2.csv",
        repetitions=[0, 1, 2, 3, 4],
        Explanation=1,
        Example=1,
        Counterexample=1,
        **{"Array/List(1/-1)": 1},
        **{"Tag/Adjacency(1/-1)": 1},
        accuracy=[0.90, 0.91, 0.90, 0.92, 0.91],
        precision=[0.35, 0.37, 0.36, 0.38, 0.35],
        recall=[0.08, 0.09, 0.08, 0.10, 0.09],
    )
    _write_evaluated(
        results_root / "algo1" / "gpt-5" / "evaluated" / "metrics_sg3_sg1.csv",
        repetitions=[0, 1, 2, 3, 4],
        Explanation=-1,
        Example=-1,
        Counterexample=-1,
        **{"Array/List(1/-1)": 1},
        **{"Tag/Adjacency(1/-1)": 1},
        accuracy=[0.50, 0.51, 0.50, 0.52, 0.51],
        precision=[0.20, 0.21, 0.20, 0.22, 0.20],
        recall=[0.05, 0.06, 0.05, 0.07, 0.05],
    )

    write_figures_bundle(results_root=results_root, output_dir=output_dir)

    overview = pd.read_csv(output_dir / "bundle_overview.csv")
    gpt5_acc_rows = overview[
        (overview["model"] == "gpt-5") & (overview["metric"] == "accuracy")
    ]

    # Must be exactly 1 row for (gpt-5, accuracy), not 2 (one per source file)
    assert len(gpt5_acc_rows) == 1, (
        f"Expected 1 overview row for gpt-5 accuracy, got {len(gpt5_acc_rows)}. "
        "Overview must have one row per model-metric, not one per source file."
    )


def test_write_figures_bundle_creates_distributional_summaries(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir(parents=True)
    output_dir = tmp_path / "bundle"

    # Create minimal evaluated CSVs for two algorithms and two models
    _write_evaluated(
        results_root
        / "algo1"
        / "gpt-4o"
        / "evaluated"
        / "metrics_sg1_sg2.csv",
        repetitions=[0, 1, 2, 3, 4],
        Explanation=-1,
        Example=-1,
        Counterexample=-1,
        **{"Array/List(1/-1)": 1},
        **{"Tag/Adjacency(1/-1)": 1},
        accuracy=[0.85, 0.86, 0.85, 0.87, 0.86],
        precision=[0.30, 0.32, 0.31, 0.33, 0.30],
        recall=[0.10, 0.11, 0.10, 0.12, 0.11],
    )
    _write_evaluated(
        results_root
        / "algo1"
        / "gpt-5"
        / "evaluated"
        / "metrics_sg1_sg2.csv",
        repetitions=[0, 1, 2, 3, 4],
        Explanation=1,
        Example=1,
        Counterexample=1,
        **{"Array/List(1/-1)": 1},
        **{"Tag/Adjacency(1/-1)": 1},
        accuracy=[0.90, 0.91, 0.90, 0.92, 0.91],
        precision=[0.35, 0.37, 0.36, 0.38, 0.35],
        recall=[0.08, 0.09, 0.08, 0.10, 0.09],
    )
    _write_evaluated(
        results_root
        / "algo2"
        / "gpt-4o"
        / "evaluated"
        / "metrics_sg1_sg2.csv",
        repetitions=[0, 1, 2, 3, 4],
        Explanation=-1,
        Example=-1,
        Counterexample=-1,
        **{"Array/List(1/-1)": 1},
        **{"Tag/Adjacency(1/-1)": 1},
        Convergence=-1,
        accuracy=[0.82, 0.83, 0.82, 0.84, 0.83],
        precision=[0.38, 0.40, 0.39, 0.41, 0.38],
        recall=[0.09, 0.10, 0.09, 0.11, 0.10],
    )

    write_figures_bundle(results_root=results_root, output_dir=output_dir)

    # --- bundle-level artifacts ---
    assert (output_dir / "README.md").exists()
    assert (output_dir / "bundle_manifest.csv").exists()
    assert (output_dir / "bundle_overview.csv").exists()

    # --- per-algorithm files ---
    assert (output_dir / "algo1_metric_rows.csv").exists()
    assert (output_dir / "algo2_metric_rows.csv").exists()

    # --- distributional summaries per model ---
    assert (output_dir / "algo1" / "gpt-4o" / "distributional_summary.csv").exists()
    assert (output_dir / "algo1" / "gpt-5" / "distributional_summary.csv").exists()
    assert (output_dir / "algo2" / "gpt-4o" / "distributional_summary.csv").exists()

    # --- manifest ---
    manifest = pd.read_csv(output_dir / "bundle_manifest.csv")
    assert len(manifest) > 0
    assert {"algorithm", "relative_path", "description"}.issubset(manifest.columns)

    # --- overview has distributional columns ---
    overview = pd.read_csv(output_dir / "bundle_overview.csv")
    required_cols = {
        "algorithm",
        "model",
        "metric",
        "n",
        "mean",
        "ci95_low",
        "ci95_high",
        "median",
        "q1",
        "q3",
    }
    assert required_cols.issubset(overview.columns)

    # --- gpt-5 algo1 should have higher mean accuracy than gpt-4o ---
    overview_df = overview[overview["metric"] == "accuracy"]
    gpt5_acc = float(
        overview_df[overview_df["model"] == "gpt-5"].iloc[0]["mean"]
    )
    gpt4o_acc = float(
        overview_df[overview_df["model"] == "gpt-4o"].iloc[0]["mean"]
    )
    assert gpt5_acc > gpt4o_acc

    # --- CI should not cross zero for accuracy ---
    for _, row in overview[overview["metric"] == "accuracy"].iterrows():
        ci_low = float(row["ci95_low"])
        ci_high = float(row["ci95_high"])
        assert ci_low > 0 or ci_high < 0, (
            f"CI [{ci_low}, {ci_high}] for {row['algorithm']} {row['model']} "
            "accuracy should not cross zero"
        )


def _write_evaluated(
    path: Path,
    *,
    repetitions: list[int],
    accuracy: list[float],
    precision: list[float],
    recall: list[float],
    **factor_columns: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {"Repetition": repetitions, "accuracy": accuracy, "precision": precision, "recall": recall}
    )
    for name, value in factor_columns.items():
        df[name] = value
    df["subgraph1"] = "[('A', 'B')]"
    df["subgraph2"] = "[('C', 'D')]"
    df["graph"] = "[('A', 'B'), ('C', 'D')]"
    df["Result"] = "[('A', 'B')]"
    df.to_csv(path, index=False)
