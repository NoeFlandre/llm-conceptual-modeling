from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis.plots import (
    _build_model_color_map,
    _canonical_model_label,
    _legend_model_order,
    write_revision_plots,
)


def test_write_revision_plots_writes_three_expected_plot_files(tmp_path: Path) -> None:
    results_root = tmp_path / "tracker"
    figures_root = results_root / "figure_exports"
    hypothesis_root = results_root / "hypothesis_testing"
    variability_root = results_root / "output_variability"
    figures_root.mkdir(parents=True)
    hypothesis_root.mkdir(parents=True)
    variability_root.mkdir(parents=True)

    pd.DataFrame(
        {
            "algorithm": ["algo1", "algo3"],
            "model": ["gpt-5", "gpt-5"],
            "metric": ["accuracy", "Recall"],
            "mean": [0.9, 0.1],
            "ci95_low": [0.88, 0.02],
            "ci95_high": [0.92, 0.18],
            "q1": [0.89, 0.0],
            "q3": [0.91, 0.15],
        }
    ).to_csv(figures_root / "bundle_overview.csv", index=False)
    pd.DataFrame(
        {
            "algorithm": ["algo1", "algo2"],
            "factor": ["Explanation", "Convergence"],
            "metric": ["precision", "accuracy"],
            "mean_difference_average": [0.02, 0.03],
            "significant_share": [0.5, 0.83],
        }
    ).to_csv(hypothesis_root / "bundle_overview.csv", index=False)
    pd.DataFrame(
        {
            "algorithm": ["algo1", "algo3"],
            "mean_pairwise_jaccard": [0.998, 0.077],
            "breadth_expansion_ratio": [1.00, 4.13],
        }
    ).to_csv(variability_root / "bundle_overview.csv", index=False)

    main_results_root = tmp_path / "results"
    (main_results_root / "algo1" / "gpt-5" / "evaluated").mkdir(parents=True)
    (main_results_root / "algo2" / "gpt-5" / "evaluated").mkdir(parents=True)
    (main_results_root / "algo3" / "gpt-5" / "evaluated").mkdir(parents=True)
    pd.DataFrame(
        {
            "accuracy": [0.9, 0.8, 0.85],
            "precision": [0.4, 0.5, 0.6],
            "recall": [0.2, 0.3, 0.4],
        }
    ).to_csv(
        main_results_root / "algo1" / "gpt-5" / "evaluated" / "metrics_sg1_sg2.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "accuracy": [0.7, 0.75, 0.72],
            "precision": [0.45, 0.5, 0.55],
            "recall": [0.35, 0.4, 0.45],
        }
    ).to_csv(
        main_results_root / "algo2" / "gpt-5" / "evaluated" / "metrics_sg1_sg2.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "Recall": [0.0, 0.3, 0.9],
        }
    ).to_csv(
        main_results_root / "algo3" / "gpt-5" / "evaluated" / "method3_results_evaluated_gpt5.csv",
        index=False,
    )

    output_dir = tmp_path / "plots"
    write_revision_plots(results_root=results_root, output_dir=output_dir)

    assert (output_dir / "distribution_metrics.png").exists()
    assert (output_dir / "factor_effect_summary.png").exists()
    assert (output_dir / "raw_output_variability.png").exists()
    assert (output_dir / "main_metric_spread_boxplots.png").exists()
    assert (output_dir / "main_metric_spread_violins.png").exists()


def test_write_revision_plots_supports_new_aggregated_hf_layout(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    aggregated_root = (
        results_root
        / "aggregated"
        / "algo1"
        / "mistralai__Ministral-3-8B-Instruct-2512"
        / "greedy"
    )
    aggregated_root.mkdir(parents=True)

    pd.DataFrame(
        {
            "pair_name": ["sg1_sg2", "sg1_sg2"],
            "accuracy": [0.8, 0.9],
            "precision": [0.4, 0.5],
            "recall": [0.2, 0.3],
            "Explanation": [-1, 1],
            "Example": [-1, 1],
            "Counterexample": [-1, 1],
            "Array/List(1/-1)": [-1, 1],
            "Tag/Adjacency(1/-1)": [-1, 1],
        }
    ).to_csv(aggregated_root / "evaluated.csv", index=False)
    pd.DataFrame(
        {
            "Feature": ["Explanation", "Example"],
            "accuracy": [12.0, 8.0],
            "recall": [10.0, 6.0],
            "precision": [7.0, 4.0],
        }
    ).to_csv(aggregated_root / "factorial.csv", index=False)
    pd.DataFrame(
        {
            "pair_name": ["sg1_sg2"],
            "mean_pairwise_jaccard": [0.99],
            "breadth_expansion_ratio": [1.01],
        }
    ).to_csv(aggregated_root / "output_variability.csv", index=False)

    output_dir = tmp_path / "plots"
    write_revision_plots(results_root=results_root, output_dir=output_dir)

    assert (output_dir / "distribution_metrics.png").exists()
    assert (output_dir / "factor_effect_summary.png").exists()
    assert (output_dir / "raw_output_variability.png").exists()


def test_build_model_color_map_groups_models_by_family_shades() -> None:
    models = [
        "deepseek-chat-v3.1",
        "deepseek-v3-chat-0324",
        "gemini-2.0-flash-exp",
        "google-gemini-2.5-pro",
        "gpt-5",
        "openai-gpt-4o",
    ]

    colors = _build_model_color_map(models)

    deepseek_a = colors["deepseek-chat-v3.1"]
    deepseek_b = colors["deepseek-v3-chat-0324"]
    gemini_a = colors["gemini-2.0-flash-exp"]
    gemini_b = colors["google-gemini-2.5-pro"]
    gpt_a = colors["gpt-5"]
    gpt_b = colors["openai-gpt-4o"]

    assert deepseek_a != deepseek_b
    assert gemini_a != gemini_b
    assert gpt_a != gpt_b
    assert deepseek_a[2] > deepseek_a[1] and deepseek_a[2] > deepseek_a[0]
    assert gemini_a[1] > gemini_a[0] and gemini_a[1] > gemini_a[2]
    assert gpt_a[0] > gpt_a[1] and gpt_a[0] > gpt_a[2]


def test_canonical_model_label_collapses_known_aliases() -> None:
    assert _canonical_model_label("google-gemini-2.5-pro") == "gemini-2.5-pro"
    assert _canonical_model_label("openai-gpt-4o") == "gpt-4o"
    assert _canonical_model_label("deepseek-chat-v3-0324") == "deepseek-v3-chat-0324"


def test_legend_model_order_groups_families_with_latest_second() -> None:
    models = [
        "gpt-5",
        "deepseek-chat-v3.1",
        "gemini-2.5-pro",
        "gpt-4o",
        "deepseek-v3-chat-0324",
        "gemini-2.0-flash-exp",
    ]

    assert _legend_model_order(models) == [
        "deepseek-v3-chat-0324",
        "deepseek-chat-v3.1",
        "gemini-2.0-flash-exp",
        "gemini-2.5-pro",
        "gpt-4o",
        "gpt-5",
    ]


def test_path_helpers() -> None:
    from pathlib import Path

    from llm_conceptual_modeling.analysis._path_helpers import (
        _discover_main_results_root,
        _path_triplet,
    )

    # Test _path_triplet
    path = Path("/results/aggregated/algo1/gpt-5/greedy/evaluated.csv")
    algo, model, cond = _path_triplet(path)
    assert algo == "algo1" and model == "gpt-5" and cond == "greedy"
    # Test _discover_main_results_root with a temp path
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        discovered = _discover_main_results_root(tmp)
        assert isinstance(discovered, Path)


def test_color_mapping_functions() -> None:
    from llm_conceptual_modeling.analysis._color_mapping import (
        _build_model_color_map,
        _canonical_model_label,
        _legend_model_order,
        _model_family,
    )
    models = ["deepseek-chat-v3.1", "gpt-5", "gemini-2.5-pro"]
    colors = _build_model_color_map(models)
    assert len(colors) == 3
    assert _canonical_model_label("google-gemini-2.5-pro") == "gemini-2.5-pro"
    assert _model_family("deepseek-v3-chat-0324") == "deepseek"
    legend_order = _legend_model_order(["gpt-5", "deepseek-v3-chat-0324"])
    assert legend_order == ["deepseek-v3-chat-0324", "gpt-5"]
