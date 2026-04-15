"""Non-LLM baseline comparison bundle.

Orchestrates the baseline comparison bundle generation and writes the
advantage summary and bundle README.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, cast

import pandas as pd

from llm_conceptual_modeling.analysis._baseline_compare import (
    _build_algo3_comparison_frame,
    _build_algo12_comparison_frame,
)
from llm_conceptual_modeling.analysis._baseline_sampling import (
    _compute_baseline_counts,
    _sample_baseline_edges,
)
from llm_conceptual_modeling.common.connection_eval import find_valid_connections
from llm_conceptual_modeling.common.types import PathLike

__all__ = [
    "_compute_baseline_counts",
    "_sample_baseline_edges",
    "find_valid_connections",
    "write_baseline_comparison_bundle",
]

_COMPARISON_BASELINE_STRATEGIES = [
    "random-k",
    "wordnet-ontology-match",
    "edit-distance",
]


def write_baseline_comparison_bundle(
    *,
    results_root: PathLike,
    output_dir: PathLike,
) -> None:
    results_root_path = Path(results_root)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    manifest_records: list[dict[str, object]] = []
    grouped_frames: list[pd.DataFrame] = []

    for algorithm in ("algo1", "algo2"):
        grouped_frame = _build_algo12_comparison_frame(
            algo=algorithm,
            results_subdir=results_root_path / algorithm,
        )
        if grouped_frame.empty:
            continue
        output_path = output_dir_path / f"{algorithm}_model_vs_baseline.csv"
        grouped_frame.to_csv(output_path, index=False)
        grouped_frames.append(grouped_frame)
        manifest_records.append(
            {
                "file": output_path.name,
                "description": (
                    f"{algorithm.upper()} per-model comparison against random-k, "
                    "WordNet-based, and edit-distance baselines."
                ),
            }
        )

    algo3_grouped = _build_algo3_comparison_frame(results_root_path / "algo3")
    if not algo3_grouped.empty:
        output_path = output_dir_path / "algo3_model_vs_baseline.csv"
        algo3_grouped.to_csv(output_path, index=False)
        grouped_frames.append(algo3_grouped)
        manifest_records.append(
            {
                "file": output_path.name,
                "description": (
                    "ALGO3 per-model comparison against random-k, WordNet-based, "
                    "and edit-distance baselines."
                ),
            }
        )

    all_models_path = output_dir_path / "all_models_vs_baseline.csv"
    if grouped_frames:
        pd.concat(grouped_frames, ignore_index=True).to_csv(all_models_path, index=False)
        manifest_records.append(
            {
                "file": all_models_path.name,
                "description": (
                    "Combined baseline comparison across all algorithms and baseline strategies."
                ),
            }
        )

    summary_path = output_dir_path / "baseline_advantage_summary.csv"
    _write_advantage_summary(grouped_frames, summary_path)
    manifest_records.append(
        {
            "file": summary_path.name,
            "description": (
                "Cross-model summary by algorithm, baseline_strategy, and metric."
            ),
        }
    )

    manifest_path = output_dir_path / "bundle_manifest.csv"
    pd.DataFrame.from_records(manifest_records).to_csv(manifest_path, index=False)
    _write_bundle_readme(output_dir_path)


def _write_advantage_summary(grouped_frames: list[pd.DataFrame], summary_path: Path) -> None:
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


def _write_bundle_readme(output_dir: Path) -> None:
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
