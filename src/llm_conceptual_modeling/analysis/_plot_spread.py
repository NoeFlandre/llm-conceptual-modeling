from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis._color_mapping import (
    _build_model_color_map,
    _legend_model_order,
)

ModelColor = tuple[float, float, float, float]


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
    if frame.empty:
        _write_empty_spread_plot(plt, output_path, metric_order)
        return

    models = sorted(frame["model"].dropna().astype(str).unique().tolist())
    color_by_model = _build_model_color_map(models)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharey=False)
    width = 0.12

    for axis, metric in zip(axes, metric_order, strict=True):
        _plot_metric(
            axis=axis,
            frame=frame,
            metric=metric,
            models=models,
            color_by_model=color_by_model,
            plot_style=plot_style,
            width=width,
        )

    axes[0].set_ylabel("Metric value")
    _add_spread_legend(fig, plt, models, color_by_model)
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_empty_spread_plot(plt, output_path: Path, metric_order: tuple[str, ...]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), sharey=False)
    for axis, metric in zip(axes, metric_order, strict=True):
        axis.set_title(metric.capitalize())
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output_path)
    plt.close(fig)


def _plot_metric(
    *,
    axis,
    frame: pd.DataFrame,
    metric: str,
    models: list[str],
    color_by_model: dict[str, ModelColor],
    plot_style: str,
    width: float,
) -> None:
    metric_frame = frame[frame["metric"] == metric]
    algorithms = _metric_algorithms(metric_frame)
    base_positions = list(range(1, len(algorithms) + 1))
    _shade_metric_bands(axis, base_positions)
    for model_index, model in enumerate(models):
        _plot_model_distribution(
            axis=axis,
            metric_frame=metric_frame,
            algorithms=algorithms,
            base_positions=base_positions,
            model=model,
            model_index=model_index,
            model_count=len(models),
            color=color_by_model[model],
            plot_style=plot_style,
            width=width,
        )
    _format_metric_axis(axis, metric, algorithms, base_positions)


def _metric_algorithms(metric_frame: pd.DataFrame) -> list[str]:
    return [
        algorithm
        for algorithm in ("algo1", "algo2", "algo3")
        if not metric_frame[metric_frame["algorithm"] == algorithm].empty
    ]


def _shade_metric_bands(axis, base_positions: list[int]) -> None:
    for band_index, position in enumerate(base_positions):
        if band_index % 2 == 0:
            axis.axvspan(position - 0.45, position + 0.45, color="#f3f4f6", zorder=0)


def _plot_model_distribution(
    *,
    axis,
    metric_frame: pd.DataFrame,
    algorithms: list[str],
    base_positions: list[int],
    model: str,
    model_index: int,
    model_count: int,
    color: ModelColor,
    plot_style: str,
    width: float,
) -> None:
    positions, values, means, margins = _metric_model_values(
        metric_frame=metric_frame,
        algorithms=algorithms,
        base_positions=base_positions,
        model=model,
        offset=(model_index - (model_count - 1) / 2) * width,
    )
    if not values:
        return
    if plot_style == "box":
        _draw_box_distribution(axis, values, positions, color, width)
    else:
        _draw_violin_distribution(axis, values, positions, color, width)
    axis.errorbar(
        positions,
        means,
        yerr=margins,
        fmt="o",
        color=color,
        capsize=3,
        markersize=3,
    )


def _metric_model_values(
    *,
    metric_frame: pd.DataFrame,
    algorithms: list[str],
    base_positions: list[int],
    model: str,
    offset: float,
) -> tuple[list[float], list[list[float]], list[float], list[float]]:
    positions: list[float] = []
    values: list[list[float]] = []
    means: list[float] = []
    margins: list[float] = []
    for algorithm_index, algorithm in enumerate(algorithms):
        series = metric_frame[
            (metric_frame["algorithm"] == algorithm) & (metric_frame["model"] == model)
        ]["value"]
        if series.empty:
            continue
        positions.append(base_positions[algorithm_index] + offset)
        values.append(series.tolist())
        means.append(float(series.mean()))
        margins.append(0.0 if len(series) <= 1 else float(1.96 * series.sem()))
    return positions, values, means, margins


def _draw_box_distribution(axis, values, positions, color: ModelColor, width: float) -> None:
    boxplot = axis.boxplot(
        values,
        positions=positions,
        widths=width * 0.9,
        patch_artist=True,
        manage_ticks=False,
        showfliers=False,
    )
    for patch in boxplot["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    for median in boxplot["medians"]:
        median.set_color(color)


def _draw_violin_distribution(axis, values, positions, color: ModelColor, width: float) -> None:
    violin = axis.violinplot(
        values,
        positions=positions,
        widths=width * 1.1,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in violin["bodies"]:
        body.set_facecolor(color)
        body.set_alpha(0.35)
        body.set_edgecolor(color)


def _format_metric_axis(
    axis,
    metric: str,
    algorithms: list[str],
    base_positions: list[int],
) -> None:
    axis.set_xticks(base_positions)
    axis.set_xticklabels([algorithm.upper() for algorithm in algorithms])
    axis.set_title(metric.capitalize())
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.set_xlabel("Algorithm")
    if base_positions:
        axis.set_xlim(0.5, len(base_positions) + 0.5)


def _add_spread_legend(
    fig,
    plt,
    models: list[str],
    color_by_model: dict[str, ModelColor],
) -> None:
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


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
