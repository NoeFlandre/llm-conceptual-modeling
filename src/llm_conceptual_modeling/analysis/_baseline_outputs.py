from __future__ import annotations

from pathlib import Path
from typing import Iterable, cast

import pandas as pd


def write_advantage_summary(
    grouped_frames: list[pd.DataFrame],
    summary_path: Path,
) -> None:
    summary_rows: list[dict[str, object]] = []
    if not grouped_frames:
        pd.DataFrame().to_csv(summary_path, index=False)
        return

    combined_frame = pd.concat(grouped_frames, ignore_index=True)
    metric_groups = cast(
        Iterable[tuple[tuple[object, object, object], pd.DataFrame]],
        combined_frame.groupby(
            ["algorithm", "baseline_strategy", "metric"],
            dropna=False,
        ),
    )
    for (algorithm, baseline_strategy, metric), metric_frame in metric_groups:
        beating_frame = metric_frame[metric_frame["mean_delta"] > 0]
        best_frame = metric_frame.sort_values("mean_delta", ascending=False)
        worst_frame = metric_frame.sort_values("mean_delta", ascending=True)
        best_row = best_frame.iloc[0]
        worst_row = worst_frame.iloc[0]
        summary_rows.append(
            {
                "algorithm": algorithm,
                "baseline_strategy": baseline_strategy,
                "metric": metric,
                "model_count": int(len(metric_frame)),
                "models_beating_baseline": int(len(beating_frame)),
                "best_model": best_row["model"],
                "best_model_delta": float(best_row["mean_delta"]),
                "worst_model": worst_row["model"],
                "worst_model_delta": float(worst_row["mean_delta"]),
                "average_model_delta": float(metric_frame["mean_delta"].mean()),
            }
        )

    pd.DataFrame.from_records(summary_rows).to_csv(summary_path, index=False)


def write_bundle_readme(output_dir: Path) -> None:
    readme = """# Non-LLM Baseline Comparison Bundle

This directory contains the organized artifacts for the non-LLM baseline comparison revision item.

## Purpose

The reviewer asked for non-LLM comparators to contextualize the value proposition of using
LLMs despite their inherent variability. The bundle compares three deterministic strategies:

- `random-k`
- `wordnet-ontology-match`
- `edit-distance`

Each baseline proposes exactly `k` edges for each row, where `k` is the number of edges
proposed by the corresponding LLM output row.

## Layout

- `bundle_manifest.csv`
  Index of every generated file with descriptions.
- `baseline_advantage_summary.csv`
  Cross-model summary by algorithm, baseline strategy, and metric.
- `<algo>_model_vs_baseline.csv`
  Per-model grouped comparison with `baseline_strategy`, `llm_mean`,
  `baseline_mean`, and `mean_delta`.
- `all_models_vs_baseline.csv`
  Combined comparison across all algorithms and baseline strategies.

## Interpretation

A positive `mean_delta` means the LLM outperforms the named non-LLM baseline on that metric.
A negative `mean_delta` means the baseline is more effective.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
