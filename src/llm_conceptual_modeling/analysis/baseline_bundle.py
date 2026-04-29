"""Non-LLM baseline comparison bundle.

Orchestrates the baseline comparison bundle generation and writes the
advantage summary and bundle README.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis._baseline_compare import (
    _build_algo3_comparison_rows,
    _build_algo12_comparison_rows,
)
from llm_conceptual_modeling.analysis._baseline_metrics import _group_comparison_rows
from llm_conceptual_modeling.analysis._baseline_outputs import (
    write_advantage_summary,
    write_bundle_readme,
    write_per_model_summary,
)
from llm_conceptual_modeling.analysis._baseline_sampling import (
    _compute_baseline_counts,
    _sample_baseline_edges,
    _scored_connection_count,
)
from llm_conceptual_modeling.common.connection_eval import find_valid_connections
from llm_conceptual_modeling.common.types import PathLike

__all__ = [
    "_compute_baseline_counts",
    "_sample_baseline_edges",
    "_scored_connection_count",
    "find_valid_connections",
    "write_baseline_comparison_bundle",
]

_COMPARISON_BASELINE_STRATEGIES = [
    "random-k",
    "wordnet-ontology-match",
]


def write_baseline_comparison_bundle(
    *,
    results_root: PathLike,
    output_dir: PathLike,
    random_repetitions: int = 5,
) -> None:
    results_root_path = Path(results_root)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    for stale_file in ["deepseek_gpt_gemini_wordnet_randomk.tex"]:
        stale_path = output_dir_path / stale_file
        if stale_path.exists():
            stale_path.unlink()

    manifest_records: list[dict[str, object]] = []
    grouped_frames: list[pd.DataFrame] = []
    row_frames: list[pd.DataFrame] = []

    for algorithm in ("algo1", "algo2"):
        comparison_rows = _build_algo12_comparison_rows(
            algo=algorithm,
            results_subdir=results_root_path / algorithm,
            random_repetitions=random_repetitions,
        )
        if not comparison_rows:
            continue
        row_frames.append(pd.DataFrame.from_records(comparison_rows))
        grouped_frame = _group_comparison_rows(comparison_rows)
        output_path = output_dir_path / f"{algorithm}_model_vs_baseline.csv"
        grouped_frame.to_csv(output_path, index=False)
        grouped_frames.append(grouped_frame)
        manifest_records.append(
            {
                "file": output_path.name,
                "description": (
                    f"{algorithm.upper()} per-model comparison against random-k, "
                    "and WordNet-based baselines."
                ),
            }
        )

    algo3_rows = _build_algo3_comparison_rows(
        results_root_path / "algo3",
        random_repetitions=random_repetitions,
    )
    if algo3_rows:
        row_frames.append(pd.DataFrame.from_records(algo3_rows))
    algo3_grouped = _group_comparison_rows(algo3_rows) if algo3_rows else pd.DataFrame()
    if not algo3_grouped.empty:
        output_path = output_dir_path / "algo3_model_vs_baseline.csv"
        algo3_grouped.to_csv(output_path, index=False)
        grouped_frames.append(algo3_grouped)
        manifest_records.append(
            {
                "file": output_path.name,
                "description": (
                    "ALGO3 per-model comparison against random-k and WordNet-based baselines."
                ),
            }
        )

    row_level_path = output_dir_path / "row_level_baseline_comparison.csv"
    if row_frames:
        pd.concat(row_frames, ignore_index=True).to_csv(row_level_path, index=False)
        manifest_records.append(
            {
                "file": row_level_path.name,
                "description": (
                    "Auditable row-level comparison including scored k and random repetitions."
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

    per_model_path = output_dir_path / "per_model_baseline_summary.csv"
    write_per_model_summary(grouped_frames, per_model_path)
    manifest_records.append(
        {
            "file": per_model_path.name,
            "description": (
                "Manuscript-facing per-model summary with random-k intervals."
            ),
        }
    )

    summary_path = output_dir_path / "baseline_advantage_summary.csv"
    write_advantage_summary(grouped_frames, summary_path)
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
    write_bundle_readme(output_dir_path)
