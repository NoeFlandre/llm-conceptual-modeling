from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd

from llm_conceptual_modeling.analysis._color_mapping import (
    _build_model_color_map,
    _canonical_model_label,
    _legend_model_order,
)
from llm_conceptual_modeling.analysis._path_helpers import (
    _discover_main_results_root,
    _path_triplet,
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

    _write_distribution_plot(figures, output_dir_path / "distribution_metrics.png")
    _write_factor_effect_plot(hypothesis, output_dir_path / "factor_effect_summary.png")
    _write_variability_plot(variability, output_dir_path / "raw_output_variability.png")
    _write_main_metric_spread_plot(
        main_metric_rows,
        output_dir_path / "main_metric_spread_boxplots.png",
        plot_style="box",
    )
    _write_main_metric_spread_plot(
        main_metric_rows,
        output_dir_path / "main_metric_spread_violins.png",
        plot_style="violin",
    )


def _build_aggregated_distribution_frame(results_root: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    evaluated_paths = list(results_root.glob("aggregated/*/*/combined/evaluated.csv"))
    if not evaluated_paths:
        evaluated_paths = list(results_root.glob("aggregated/*/*/*/evaluated.csv"))
    for evaluated_path in evaluated_paths:
        algorithm, model, _condition = _path_triplet(evaluated_path)
        frame = pd.read_csv(evaluated_path)
        metric_columns = [
            column for column in ("accuracy", "precision", "recall", "Recall") if column in frame
        ]
        for metric in metric_columns:
            series = frame[metric]
            for value in series.tolist():
                records.append(
                    {
                        "algorithm": algorithm,
                        "model": model,
                        "metric": metric,
                        "value": float(value),
                    }
                )
    return pd.DataFrame.from_records(records)


def _build_aggregated_factor_effect_frame(results_root: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    factorial_paths = list(results_root.glob("aggregated/*/*/combined/factorial.csv"))
    if not factorial_paths:
        factorial_paths = list(results_root.glob("aggregated/*/*/*/factorial.csv"))
    for factorial_path in factorial_paths:
        algorithm, model, _condition = _path_triplet(factorial_path)
        if algorithm not in {"algo1", "algo2"}:
            continue
        frame = pd.read_csv(factorial_path)
        feature_column = "Feature" if "Feature" in frame.columns else "feature"
        metric_columns = [
            column for column in ("accuracy", "precision", "recall") if column in frame.columns
        ]
        filtered = frame[
            ~frame[feature_column].isin(["Error", "Repetition"])
            & ~frame[feature_column].astype(str).str.contains("_AND_", regex=False)
        ]
        for _, row in filtered.iterrows():
            for metric in metric_columns:
                records.append(
                    {
                        "algorithm": algorithm,
                        "model": model,
                        "factor": row[feature_column],
                        "metric": metric,
                        "effect_share": float(row[metric]),
                    }
                )
    if not records:
        return pd.DataFrame(
            columns=[
                "algorithm",
                "factor",
                "metric",
                "effect_share",
            ]
        )
    return pd.DataFrame.from_records(records)


def _build_aggregated_variability_frame(results_root: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    variability_paths = list(results_root.glob("aggregated/*/*/combined/output_variability.csv"))
    if not variability_paths:
        variability_paths = list(results_root.glob("aggregated/*/*/*/output_variability.csv"))
    for variability_path in variability_paths:
        algorithm, model, _condition = _path_triplet(variability_path)
        frame = pd.read_csv(variability_path)
        if frame.empty:
            continue
        records.append(
            {
                "algorithm": algorithm,
                "model": model,
                "mean_pairwise_jaccard": float(frame["mean_pairwise_jaccard"].mean()),
                "breadth_expansion_ratio": float(
                    frame["union_edge_count"].mean()
                    / frame["mean_edge_count"].clip(lower=1.0).mean()
                )
                if "union_edge_count" in frame and "mean_edge_count" in frame
                else float(frame["breadth_expansion_ratio"].mean()),
            }
        )
    return pd.DataFrame.from_records(records)


def _build_main_metric_rows(results_root: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    metric_map = {
        "algo1": ("accuracy", "precision", "recall"),
        "algo2": ("accuracy", "precision", "recall"),
        "algo3": ("Recall",),
    }
    for algorithm, metrics in metric_map.items():
        for evaluated_path in sorted((results_root / algorithm).glob("*/evaluated/*.csv")):
            model = _canonical_model_label(evaluated_path.parts[-3])
            frame = pd.read_csv(evaluated_path)
            for metric in metrics:
                if metric not in frame.columns:
                    continue
                normalized_metric = metric.lower()
                for value in frame[metric].dropna().tolist():
                    records.append(
                        {
                            "algorithm": algorithm,
                            "model": model,
                            "metric": normalized_metric,
                            "value": float(value),
                        }
                    )
    return pd.DataFrame.from_records(records)


def _write_distribution_plot(frame: pd.DataFrame, output_path: Path) -> None:
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    if frame.empty:
        ax.set_title("Main metric distributions by algorithm and model")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return
    if "value" not in frame.columns:
        labels = [
            f"{cast(str, row['algorithm'])}\n{cast(str, row['model'])}\n{cast(str, row['metric'])}"
            for _, row in frame.iterrows()
        ]
        means = frame["mean"].tolist()
        yerr_low = frame["mean"] - frame["ci95_low"]
        yerr_high = frame["ci95_high"] - frame["mean"]
        ax.errorbar(range(len(means)), means, yerr=[yerr_low, yerr_high], fmt="o", capsize=4)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Metric value")
        ax.set_title("Main metric distributions by algorithm and model")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return
    grouped = list(frame.groupby(["algorithm", "model", "metric"], dropna=False))
    labels = [
        f"{cast(tuple[object, object, object], keys)[0]}\n"
        f"{cast(tuple[object, object, object], keys)[1]}\n"
        f"{cast(tuple[object, object, object], keys)[2]}"
        for keys, _group in grouped
    ]
    values = [group["value"].tolist() for _keys, group in grouped]
    positions = list(range(1, len(values) + 1))
    ax.boxplot(values, positions=positions, widths=0.6, patch_artist=True)
    means: list[float] = []
    yerr_low: list[float] = []
    yerr_high: list[float] = []
    for _keys, group in grouped:
        series = group["value"]
        mean = float(series.mean())
        margin = 0.0 if len(series) <= 1 else float(1.96 * series.sem())
        means.append(mean)
        yerr_low.append(margin)
        yerr_high.append(margin)
    ax.errorbar(positions, means, yerr=[yerr_low, yerr_high], fmt="o", color="black", capsize=4)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("Main metric distributions by algorithm and model")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


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
