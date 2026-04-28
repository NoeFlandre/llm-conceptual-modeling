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


def write_per_model_summary(
    grouped_frames: list[pd.DataFrame],
    output_path: Path,
) -> None:
    if not grouped_frames:
        pd.DataFrame().to_csv(output_path, index=False)
        return

    combined_frame = pd.concat(grouped_frames, ignore_index=True)
    index_columns = ["algorithm", "model", "metric"]
    base_columns = [*index_columns, "llm_mean", "mean_k", "row_count"]
    row_count_source = "llm_row_count" if "llm_row_count" in combined_frame.columns else "row_count"
    summary = (
        combined_frame[index_columns + ["llm_mean", "mean_k", row_count_source]]
        .groupby(index_columns, dropna=False)
        .agg(
            llm_mean=("llm_mean", "first"),
            mean_k=("mean_k", "mean"),
            row_count=(row_count_source, "max"),
        )
        .reset_index()
    )

    for strategy in sorted(combined_frame["baseline_strategy"].dropna().unique()):
        strategy_frame = combined_frame[combined_frame["baseline_strategy"] == strategy]
        prefix = str(strategy).replace("-", "_")
        columns = index_columns + [
            "baseline_mean",
            "baseline_ci95_low",
            "baseline_ci95_high",
            "mean_delta",
        ]
        renamed = strategy_frame[columns].rename(
            columns={
                "baseline_mean": f"{prefix}_mean",
                "baseline_ci95_low": f"{prefix}_ci95_low",
                "baseline_ci95_high": f"{prefix}_ci95_high",
                "mean_delta": f"{prefix}_mean_delta",
            }
        )
        summary = summary.merge(renamed, on=index_columns, how="left")

    summary[[column for column in base_columns if column in summary.columns] + [
        column for column in summary.columns if column not in base_columns
    ]].to_csv(output_path, index=False)


def write_bundle_readme(output_dir: Path) -> None:
    readme = """# Non-LLM Baseline Comparison Bundle

This directory contains the organized artifacts for the non-LLM baseline comparison revision item.

## Purpose

The reviewer asked for non-LLM comparators to contextualize the value proposition of using
LLMs despite their inherent variability. The bundle compares three baseline strategies:

- `random-k`, sampled from all admissible cross-subgraph pairs with five deterministic replications.
- `wordnet-ontology-match`, a volume-matched direct lexical matching baseline.
- `edit-distance`

Each baseline proposes exactly `k` edges for each row, where `k` is the number of
scored cross-subgraph connections produced by the corresponding LLM output row after adding its
generated edges to the source and target graphs. This avoids rewarding a method for simply
emitting more raw edges when only a smaller number of source-target connections are scored.

## Layout

- `bundle_manifest.csv`
  Index of every generated file with descriptions.
- `row_level_baseline_comparison.csv`
  Auditable row-level comparison with the scored `k`, random-k repetition index, LLM metric,
  baseline metric, and delta.
- `per_model_baseline_summary.csv`
  Manuscript-facing per-model summary. Use this file when asking whether a top LLM can
  outperform a baseline rather than relying only on a cross-model frontier mean.
- `baseline_advantage_summary.csv`
  Cross-model summary by algorithm, baseline strategy, and metric.
- `<algo>_model_vs_baseline.csv`
  Per-model grouped comparison with `baseline_strategy`, `llm_mean`,
  `baseline_mean`, confidence interval columns for the baseline mean, and `mean_delta`.
- `all_models_vs_baseline.csv`
  Combined comparison across all algorithms and baseline strategies.

## Interpretation

A positive `mean_delta` means the LLM outperforms the named non-LLM baseline on that metric.
A negative `mean_delta` means the baseline is more effective.
WordNet should be interpreted as a clean direct lexical matching baseline: it does not propose
new intermediate concept nodes and therefore does not solve the same generative task as the LLM
methods.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
