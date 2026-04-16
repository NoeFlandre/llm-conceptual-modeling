from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_factor_effect_plot(frame: pd.DataFrame, output_path: Path) -> None:
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


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
