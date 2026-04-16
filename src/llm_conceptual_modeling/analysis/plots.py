from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd

from llm_conceptual_modeling.analysis._path_helpers import _discover_main_results_root
from llm_conceptual_modeling.analysis._plot_distribution import (
    write_distribution_plot,
)
from llm_conceptual_modeling.analysis._plot_frames import (
    _build_aggregated_distribution_frame,
    _build_aggregated_factor_effect_frame,
    _build_aggregated_variability_frame,
    _build_main_metric_rows,
)
from llm_conceptual_modeling.analysis._plot_spread import (
    write_main_metric_spread_plots,
)
from llm_conceptual_modeling.common.types import PathLike


def write_revision_plots(*, results_root: PathLike, output_dir: PathLike) -> None:
    results_root_path = Path(results_root)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    if (results_root_path / "figure_exports" / "bundle_overview.csv").exists():
        figures = pd.read_csv(results_root_path / "figure_exports" / "bundle_overview.csv")
        hypothesis = pd.read_csv(results_root_path / "hypothesis_testing" / "bundle_overview.csv")
        variability = pd.read_csv(results_root_path / "output_variability" / "bundle_overview.csv")
    else:
        figures = _build_aggregated_distribution_frame(results_root_path)
        hypothesis = _build_aggregated_factor_effect_frame(results_root_path)
        variability = _build_aggregated_variability_frame(results_root_path)
    main_metric_rows = _build_main_metric_rows(_discover_main_results_root(results_root_path))

    write_distribution_plot(figures, output_dir_path / "distribution_metrics.png")
    _write_factor_effect_plot(hypothesis, output_dir_path / "factor_effect_summary.png")
    _write_variability_plot(variability, output_dir_path / "raw_output_variability.png")
    write_main_metric_spread_plots(
        frame=main_metric_rows,
        boxplot_output_path=output_dir_path / "main_metric_spread_boxplots.png",
        violin_output_path=output_dir_path / "main_metric_spread_violins.png",
    )


def _write_factor_effect_plot(frame: pd.DataFrame, output_path: Path) -> None:
    plt = _load_pyplot()
    if frame.empty:
        fig, ax = plt.subplots(figsize=(9, 5.2))
        ax.set_title("Factor-effect heatmap")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return
    figure_data = frame.copy()
    if (
        "effect_share" not in figure_data.columns
        and "mean_difference_average" in figure_data.columns
    ):
        figure_data["effect_share"] = figure_data["mean_difference_average"]
    figure_data["column_label"] = (
        figure_data["algorithm"].astype(str) + ":" + figure_data["metric"].astype(str)
    )
    pivot = figure_data.pivot_table(
        index="factor",
        columns="column_label",
        values="effect_share",
        aggfunc="mean",
        fill_value=0.0,
    )
    fig, ax = plt.subplots(figsize=(9, 5.2))
    image = ax.imshow(pivot.to_numpy(), cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Factor-effect heatmap")
    fig.colorbar(image, ax=ax, label="Effect share (%)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _write_variability_plot(frame: pd.DataFrame, output_path: Path) -> None:
    plt = _load_pyplot()
    if frame.empty:
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        ax.set_title("Raw-output variability by algorithm and model")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharex=True)
    labels = [
        (
            f"{cast(str, row['algorithm'])}\n{cast(str, row['model'])}"
            if "model" in frame.columns
            else cast(str, row["algorithm"])
        )
        for _, row in frame.iterrows()
    ]
    positions = range(len(labels))
    axes[0].bar(positions, frame["mean_pairwise_jaccard"].tolist(), color="#1f77b4")
    axes[0].set_title("Pairwise Jaccard")
    axes[0].set_ylabel("Mean overlap")
    axes[1].bar(positions, frame["breadth_expansion_ratio"].tolist(), color="#d62728")
    axes[1].set_title("Breadth-expansion ratio")
    axes[1].set_ylabel("Union / mean edge count")
    for axis in axes:
        axis.set_xticks(list(positions))
        axis.set_xticklabels(labels, rotation=45, ha="right")
    fig.suptitle("Raw-output variability by algorithm and model")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
