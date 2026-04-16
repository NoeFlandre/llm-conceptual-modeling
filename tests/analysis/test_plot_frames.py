from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis._plot_frames import (
    _build_aggregated_distribution_frame,
    _build_aggregated_factor_effect_frame,
    _build_aggregated_variability_frame,
    _build_main_metric_rows,
)


def test_build_aggregated_plot_frames(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    aggregated_root = (
        results_root / "aggregated" / "algo1" / "gpt-5" / "greedy"
    )
    aggregated_root.mkdir(parents=True)

    pd.DataFrame(
        {
            "pair_name": ["sg1_sg2"],
            "accuracy": [0.8],
            "precision": [0.4],
            "recall": [0.2],
            "Explanation": [-1],
            "Example": [1],
            "Counterexample": [1],
            "Array/List(1/-1)": [-1],
            "Tag/Adjacency(1/-1)": [1],
        }
    ).to_csv(aggregated_root / "evaluated.csv", index=False)
    pd.DataFrame(
        {
            "Feature": ["Explanation"],
            "accuracy": [12.0],
            "recall": [10.0],
            "precision": [7.0],
        }
    ).to_csv(aggregated_root / "factorial.csv", index=False)
    pd.DataFrame(
        {
            "pair_name": ["sg1_sg2"],
            "mean_pairwise_jaccard": [0.99],
            "breadth_expansion_ratio": [1.01],
        }
    ).to_csv(aggregated_root / "output_variability.csv", index=False)

    for algorithm in ("algo1", "algo2", "algo3"):
        evaluated_root = results_root / algorithm / "gpt-5" / "evaluated"
        evaluated_root.mkdir(parents=True)
    pd.DataFrame(
        {
            "accuracy": [0.9],
            "precision": [0.4],
            "recall": [0.2],
        }
    ).to_csv(
        results_root / "algo1" / "gpt-5" / "evaluated" / "metrics_sg1_sg2.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "accuracy": [0.7],
            "precision": [0.5],
            "recall": [0.3],
        }
    ).to_csv(
        results_root / "algo2" / "gpt-5" / "evaluated" / "metrics_sg1_sg2.csv",
        index=False,
    )
    pd.DataFrame({"Recall": [0.1]}).to_csv(
        results_root / "algo3" / "gpt-5" / "evaluated" / "method3_results_evaluated_gpt5.csv",
        index=False,
    )

    distribution = _build_aggregated_distribution_frame(results_root)
    factor_effect = _build_aggregated_factor_effect_frame(results_root)
    variability = _build_aggregated_variability_frame(results_root)
    main_rows = _build_main_metric_rows(results_root)

    assert distribution["metric"].tolist() == ["accuracy", "precision", "recall"]
    assert factor_effect["factor"].unique().tolist() == ["Explanation"]
    assert variability["mean_pairwise_jaccard"].tolist() == [0.99]
    assert set(main_rows["metric"].tolist()) == {"accuracy", "precision", "recall"}
    assert main_rows[main_rows["algorithm"] == "algo3"]["metric"].tolist() == ["recall"]
