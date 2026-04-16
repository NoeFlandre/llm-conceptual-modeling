"""Non-LLM baseline comparison bundle.

Orchestrates the baseline comparison bundle generation and writes the
advantage summary and bundle README.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis._baseline_compare import (
    _build_algo3_comparison_frame,
    _build_algo12_comparison_frame,
)
from llm_conceptual_modeling.analysis._baseline_outputs import (
    write_advantage_summary,
    write_bundle_readme,
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
