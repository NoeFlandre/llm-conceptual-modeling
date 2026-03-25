from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis.figures_bundle import write_figures_bundle


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
