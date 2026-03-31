from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd

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

    _write_distribution_plot(figures, output_dir_path / "distribution_metrics.png")
    _write_factor_effect_plot(hypothesis, output_dir_path / "factor_effect_summary.png")
    _write_variability_plot(variability, output_dir_path / "raw_output_variability.png")


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


def _path_triplet(path: Path) -> tuple[str, str, str]:
    parts = path.parts
    aggregated_index = parts.index("aggregated")
    return (
        parts[aggregated_index + 1],
        parts[aggregated_index + 2],
        parts[aggregated_index + 3],
    )


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
