"""Baseline comparison bundle for open-weight map-extension ALGO3 runs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis._baseline_compare import (
    _COMPARISON_BASELINE_STRATEGIES,
    _baseline_repetitions,
    _baseline_seed,
)
from llm_conceptual_modeling.analysis._baseline_metrics import (
    _build_algo3_metric_rows,
    _group_comparison_rows,
)
from llm_conceptual_modeling.analysis._baseline_outputs import write_per_model_summary
from llm_conceptual_modeling.analysis._baseline_sampling import (
    _sample_baseline_edges,
    _scored_connection_count,
)
from llm_conceptual_modeling.analysis._edge_parsing import _parse_algo3_edge_list
from llm_conceptual_modeling.common.connection_eval import find_valid_connections
from llm_conceptual_modeling.common.types import PathLike


def write_map_extension_baseline_bundle(
    *,
    results_root: PathLike,
    output_dir: PathLike,
    random_repetitions: int = 5,
) -> None:
    results_root_path = Path(results_root)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    comparison_rows = _build_map_extension_comparison_rows(
        results_root_path,
        random_repetitions=random_repetitions,
    )
    row_level_path = output_dir_path / "row_level_baseline_comparison.csv"
    grouped_path = output_dir_path / "map_extension_model_vs_baseline.csv"
    all_models_path = output_dir_path / "all_models_vs_baseline.csv"
    per_model_path = output_dir_path / "per_model_baseline_summary.csv"

    if comparison_rows:
        row_frame = pd.DataFrame.from_records(comparison_rows)
        row_frame.to_csv(row_level_path, index=False)
        grouped_frame = _group_comparison_rows(
            comparison_rows,
            group_columns=[
                "algorithm",
                "graph_source",
                "model",
                "baseline_strategy",
                "metric",
            ],
        )
        grouped_frame.to_csv(grouped_path, index=False)
        all_models_frame = _group_comparison_rows(comparison_rows)
        all_models_frame.to_csv(all_models_path, index=False)
        write_per_model_summary([all_models_frame], per_model_path)
    else:
        pd.DataFrame().to_csv(row_level_path, index=False)
        pd.DataFrame().to_csv(grouped_path, index=False)
        pd.DataFrame().to_csv(all_models_path, index=False)
        pd.DataFrame().to_csv(per_model_path, index=False)

    manifest = pd.DataFrame.from_records(
        [
            {
                "file": row_level_path.name,
                "description": "Auditable row-level ALGO3 map-extension baseline comparison.",
            },
            {
                "file": grouped_path.name,
                "description": "Grouped map-extension baseline comparison by case study and model.",
            },
            {
                "file": per_model_path.name,
                "description": "Per-model map-extension baseline summary.",
            },
            {
                "file": all_models_path.name,
                "description": "Grouped map-extension baseline comparison across case studies.",
            },
        ]
    )
    manifest.to_csv(output_dir_path / "bundle_manifest.csv", index=False)
    _write_map_extension_readme(output_dir_path)


def _build_map_extension_comparison_rows(
    results_root: Path,
    *,
    random_repetitions: int,
) -> list[dict[str, object]]:
    batch_summary_path = results_root / "batch_summary.csv"
    if not batch_summary_path.exists():
        return []

    batch_summary = pd.read_csv(batch_summary_path)
    comparison_rows: list[dict[str, object]] = []
    for row_position, (_, summary_row) in enumerate(batch_summary.iterrows()):
        if str(summary_row.get("status", "")).lower() != "finished":
            continue
        raw_row_path = _resolve_raw_row_path(
            results_root,
            str(summary_row.get("raw_row_path", "")),
        )
        raw_row = json.loads(raw_row_path.read_text(encoding="utf-8"))
        source_edges = _parse_algo3_edge_list(raw_row.get("Source Graph"))
        target_edges = _parse_algo3_edge_list(raw_row.get("Target Graph"))
        mother_edges = _parse_algo3_edge_list(raw_row.get("Mother Graph"))
        llm_edges = set(_parse_algo3_edge_list(raw_row.get("Results")))
        k = _scored_connection_count(
            sorted(llm_edges),
            subgraph1_edges=source_edges,
            subgraph2_edges=target_edges,
        )
        ground_truth = set(find_valid_connections(mother_edges, source_edges, target_edges))

        model = str(raw_row.get("model") or summary_row.get("model"))
        graph_source = str(raw_row.get("graph_source") or summary_row.get("graph_source"))
        pair_name = str(raw_row.get("pair_name") or summary_row.get("pair_name"))
        decoding_algorithm = str(
            raw_row.get("decoding_algorithm") or summary_row.get("decoding_algorithm")
        )
        for baseline_strategy in _COMPARISON_BASELINE_STRATEGIES:
            for baseline_repetition in _baseline_repetitions(
                baseline_strategy,
                random_repetitions,
            ):
                random_seed = _baseline_seed(
                    "open_weight_map_extension",
                    model,
                    f"{graph_source}/{pair_name}/{decoding_algorithm}",
                    row_position,
                    baseline_strategy,
                    baseline_repetition,
                )
                baseline_edges = _sample_baseline_edges(
                    baseline_strategy=baseline_strategy,
                    k=k,
                    mother_edges=mother_edges,
                    subgraph1_edges=source_edges,
                    subgraph2_edges=target_edges,
                    random_seed=random_seed,
                )
                comparison_rows.extend(
                    row
                    for row in _build_algo3_metric_rows(
                        model=model,
                        baseline_strategy=baseline_strategy,
                        source_file=str(raw_row_path.relative_to(results_root)),
                        k=k,
                        source_row=row_position,
                        baseline_repetition=baseline_repetition,
                        llm_recall=float(
                            summary_row.get("recall", raw_row.get("Recall", 0))
                        ),
                        llm_edges=llm_edges,
                        baseline_edges=baseline_edges,
                        ground_truth=ground_truth,
                        mother_edges=mother_edges,
                        source_edges=source_edges,
                        target_edges=target_edges,
                        extra_fields={
                            "graph_source": graph_source,
                            "pair_name": pair_name,
                            "decoding_algorithm": decoding_algorithm,
                        },
                    )
                    if row["metric"] == "recall"
                )
    return comparison_rows


def _resolve_raw_row_path(results_root: Path, raw_row_path: str) -> Path:
    candidate = Path(raw_row_path)
    if candidate.exists():
        return candidate

    parts = candidate.parts
    if "runs" in parts:
        runs_index = parts.index("runs")
        resolved = results_root.joinpath(*parts[runs_index:])
        if resolved.exists():
            return resolved

    msg = f"Could not resolve raw row path under {results_root}: {raw_row_path}"
    raise FileNotFoundError(msg)


def _write_map_extension_readme(output_dir: Path) -> None:
    readme = """# Open-Weight Map-Extension Baseline Comparison Bundle

This directory contains the corrected non-LLM baseline comparison for the ALGO3
open-weight map-extension runs. Because map extension is interpreted through recall,
this bundle reports recall only. `k` is the scored LLM cross-subgraph connection count,
random-k is sampled from all admissible source-target pairs with five deterministic
replications, and WordNet is interpreted as a direct lexical matching baseline.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
