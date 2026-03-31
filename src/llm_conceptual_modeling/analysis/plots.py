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
    for evaluated_path in results_root.glob("aggregated/*/*/*/evaluated.csv"):
        algorithm, model, _condition = _path_triplet(evaluated_path)
        frame = pd.read_csv(evaluated_path)
        metric_columns = [
            column for column in ("accuracy", "precision", "recall", "Recall") if column in frame
        ]
        for metric in metric_columns:
            series = frame[metric]
            if series.empty:
                continue
            mean = float(series.mean())
            margin = 0.0 if len(series) <= 1 else float(1.96 * series.sem())
            records.append(
                {
                    "algorithm": algorithm,
                    "model": model,
                    "metric": metric,
                    "mean": mean,
                    "ci95_low": mean - margin,
                    "ci95_high": mean + margin,
                    "q1": float(series.quantile(0.25)),
                    "q3": float(series.quantile(0.75)),
                }
            )
    return pd.DataFrame.from_records(records)


def _build_aggregated_factor_effect_frame(results_root: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for evaluated_path in results_root.glob("aggregated/*/*/*/evaluated.csv"):
        algorithm, model, condition_label = _path_triplet(evaluated_path)
        if algorithm not in {"algo1", "algo2"}:
            continue
        frame = pd.read_csv(evaluated_path)
        factor_columns = [
            column
            for column in (
                "Explanation",
                "Example",
                "Counterexample",
                "Array/List(1/-1)",
                "Tag/Adjacency(1/-1)",
            "Convergence",
        )
        if column in frame
        ]
        metric_columns = [
            column for column in ("accuracy", "precision", "recall") if column in frame
        ]
        for factor in factor_columns:
            low_frame = frame[frame[factor] == -1]
            high_frame = frame[frame[factor] == 1]
            if low_frame.empty or high_frame.empty:
                continue
            for metric in metric_columns:
                records.append(
                    {
                        "algorithm": algorithm,
                        "model": model,
                        "condition_label": condition_label,
                        "factor": factor,
                        "metric": metric,
                        "mean_difference_average": float(
                            high_frame[metric].mean() - low_frame[metric].mean()
                        ),
                    }
                )
    if not records:
        return pd.DataFrame(
            columns=[
                "algorithm",
                "factor",
                "metric",
                "mean_difference_average",
                "significant_share",
            ]
        )
    frame = pd.DataFrame.from_records(records)
    grouped = (
        frame.groupby(["algorithm", "factor", "metric"], dropna=False)["mean_difference_average"]
        .mean()
        .reset_index()
    )
    grouped["significant_share"] = 1.0
    return grouped


def _build_aggregated_variability_frame(results_root: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for variability_path in results_root.glob("aggregated/*/*/*/output_variability.csv"):
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
    fig, ax = plt.subplots(figsize=(9, 4.8))
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
    ax.set_title("Main metric distributions with mean and 95% CI")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _write_factor_effect_plot(frame: pd.DataFrame, output_path: Path) -> None:
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = [
        f"{cast(str, row['algorithm'])}:{cast(str, row['factor'])}\n{cast(str, row['metric'])}"
        for _, row in frame.iterrows()
    ]
    values = frame["mean_difference_average"].tolist()
    colors = ["#1f77b4" if value >= 0 else "#d62728" for value in values]
    ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("High minus low level mean")
    ax.set_title("Factor-effect summary")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _write_variability_plot(frame: pd.DataFrame, output_path: Path) -> None:
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.scatter(
        frame["mean_pairwise_jaccard"],
        frame["breadth_expansion_ratio"],
        s=80,
    )
    for _, row in frame.iterrows():
        algorithm = cast(str, row["algorithm"])
        model = cast(str | None, row.get("model"))
        label = algorithm if not model else f"{algorithm}:{model}"
        ax.annotate(
            label,
            (
                cast(float, row["mean_pairwise_jaccard"]),
                cast(float, row["breadth_expansion_ratio"]),
            ),
        )
    ax.set_xlabel("Mean pairwise Jaccard")
    ax.set_ylabel("Breadth-expansion ratio")
    ax.set_title("Raw-output variability by algorithm")
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
