from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd


def write_variability_plot(frame: pd.DataFrame, output_path: Path) -> None:
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
