from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd


def write_distribution_plot(frame: pd.DataFrame, output_path: Path) -> None:
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


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
