from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis._color_mapping import (
    _build_model_color_map,
    _legend_model_order,
)


def write_main_metric_spread_plots(
    *,
    frame: pd.DataFrame,
    boxplot_output_path: Path,
    violin_output_path: Path,
) -> None:
    _write_main_metric_spread_plot(
        frame,
        boxplot_output_path,
        plot_style="box",
    )
    _write_main_metric_spread_plot(
        frame,
        violin_output_path,
        plot_style="violin",
    )


def _write_main_metric_spread_plot(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    plot_style: str,
) -> None:
    plt = _load_pyplot()
    metric_order = ("accuracy", "precision", "recall")
    algorithm_order = ("algo1", "algo2", "algo3")
    if frame.empty:
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), sharey=False)
        for axis, metric in zip(axes, metric_order, strict=False):
            axis.set_title(metric.capitalize())
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        fig.savefig(output_path)
        plt.close(fig)
        return

    models = sorted(frame["model"].dropna().astype(str).unique().tolist())
    color_by_model = _build_model_color_map(models)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharey=False)
    width = 0.12

    for axis, metric in zip(axes, metric_order, strict=False):
        metric_frame = frame[frame["metric"] == metric]
        algorithms_for_metric = [
            algorithm
            for algorithm in algorithm_order
            if not metric_frame[metric_frame["algorithm"] == algorithm].empty
        ]
        base_positions = list(range(1, len(algorithms_for_metric) + 1))
        for band_index, position in enumerate(base_positions):
            if band_index % 2 == 0:
                axis.axvspan(position - 0.45, position + 0.45, color="#f3f4f6", zorder=0)
        for model_index, model in enumerate(models):
            offset = (model_index - (len(models) - 1) / 2) * width
            positions: list[float] = []
            values: list[list[float]] = []
            means: list[float] = []
            margins: list[float] = []
            for algorithm_index, algorithm in enumerate(algorithms_for_metric):
                series = metric_frame[
                    (metric_frame["algorithm"] == algorithm) & (metric_frame["model"] == model)
                ]["value"]
                if series.empty:
                    continue
                position = base_positions[algorithm_index] + offset
                positions.append(position)
                values.append(series.tolist())
                mean = float(series.mean())
                margin = 0.0 if len(series) <= 1 else float(1.96 * series.sem())
                means.append(mean)
                margins.append(margin)
            if not values:
                continue
            if plot_style == "box":
                boxplot = axis.boxplot(
                    values,
                    positions=positions,
                    widths=width * 0.9,
                    patch_artist=True,
                    manage_ticks=False,
                    showfliers=False,
                )
                for patch in boxplot["boxes"]:
                    patch.set_facecolor(color_by_model[model])
                    patch.set_alpha(0.5)
                for median in boxplot["medians"]:
                    median.set_color(color_by_model[model])
            else:
                violin = axis.violinplot(
                    values,
                    positions=positions,
                    widths=width * 1.1,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                )
                for body in violin["bodies"]:
                    body.set_facecolor(color_by_model[model])
                    body.set_alpha(0.35)
                    body.set_edgecolor(color_by_model[model])
            axis.errorbar(
                positions,
                means,
                yerr=margins,
                fmt="o",
                color=color_by_model[model],
                capsize=3,
                markersize=3,
            )
        axis.set_xticks(base_positions)
        axis.set_xticklabels([algorithm.upper() for algorithm in algorithms_for_metric])
        axis.set_title(metric.capitalize())
        axis.set_ylim(bottom=0.0)
        axis.grid(axis="y", alpha=0.25)
        axis.set_xlabel("Algorithm")
        if base_positions:
            axis.set_xlim(0.5, len(base_positions) + 0.5)

    axes[0].set_ylabel("Metric value")
    legend_models = _legend_model_order(models)
    handles = [
        plt.Line2D([0], [0], color=color_by_model[model], lw=6, alpha=0.6)
        for model in legend_models
    ]
    fig.legend(
        handles,
        legend_models,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
