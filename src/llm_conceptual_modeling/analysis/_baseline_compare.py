"""Comparison frame builders for baseline comparison.

Extracts the frame-building logic that walks model directories and constructs
comparison rows for both algo1/2 and algo3 evaluation flows.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis._baseline_metrics import (
    _build_algo3_metric_rows,
    _build_algo12_metric_rows,
    _group_comparison_rows,
)
from llm_conceptual_modeling.analysis._baseline_sampling import (
    _compute_baseline_counts,
    _sample_baseline_edges,
)
from llm_conceptual_modeling.analysis._edge_parsing import (
    _parse_algo3_edge_list,
    _parse_edges,
)
from llm_conceptual_modeling.common.connection_eval import find_valid_connections

_COMPARISON_BASELINE_STRATEGIES = [
    "random-k",
    "wordnet-ontology-match",
    "edit-distance",
]


def _raw_from_evaluated(eval_file: Path, model_dir: Path) -> Path | None:
    eval_name = eval_file.name.replace("metrics_", "").replace(".csv", "")
    raw_dir = model_dir / "raw"
    if not raw_dir.is_dir():
        return None
    for raw_file in sorted(raw_dir.glob("*.csv")):
        if eval_name in raw_file.name:
            return raw_file
    return None


def _build_algo12_comparison_frame(
    *,
    algo: str,
    results_subdir: Path,
) -> pd.DataFrame:
    comparison_rows: list[dict[str, object]] = []
    if not results_subdir.is_dir():
        return pd.DataFrame()

    for model_dir in sorted(results_subdir.iterdir()):
        eval_dir = model_dir / "evaluated"
        if not eval_dir.is_dir():
            continue
        model = model_dir.name

        for eval_file in sorted(eval_dir.glob("metrics_*.csv")):
            raw_file = _raw_from_evaluated(eval_file, model_dir)
            if raw_file is None:
                continue

            evaluated = pd.read_csv(eval_file)
            raw = pd.read_csv(raw_file)

            for row_index, row in evaluated.iterrows():
                raw_row = raw.iloc[row_index]
                llm_result_edges = _parse_edges(raw_row.get("Result", "[]"))
                k = len(llm_result_edges)

                mother_edges = _parse_edges(raw_row.get("graph", "[]"))
                subgraph1_edges = _parse_edges(raw_row.get("subgraph1", "[]"))
                subgraph2_edges = _parse_edges(raw_row.get("subgraph2", "[]"))
                ground_truth = set(
                    find_valid_connections(mother_edges, subgraph1_edges, subgraph2_edges)
                )

                for baseline_strategy in _COMPARISON_BASELINE_STRATEGIES:
                    baseline_counts = _compute_baseline_counts(
                        baseline_strategy=baseline_strategy,
                        k=k,
                        mother_edges=mother_edges,
                        subgraph1_edges=subgraph1_edges,
                        subgraph2_edges=subgraph2_edges,
                        ground_truth=ground_truth,
                    )
                    comparison_rows.extend(
                        _build_algo12_metric_rows(
                            algo=algo,
                            model=model,
                            baseline_strategy=baseline_strategy,
                            source_file=eval_file.name,
                            llm_accuracy=float(row.get("accuracy", 0)),
                            llm_precision=float(row.get("precision", 0)),
                            llm_recall=float(row.get("recall", 0)),
                            k=k,
                            baseline_tp=baseline_counts["tp"],
                            baseline_fp=baseline_counts["fp"],
                            baseline_fn=baseline_counts["fn"],
                            subgraph1_edges=subgraph1_edges,
                            subgraph2_edges=subgraph2_edges,
                        )
                    )

    if not comparison_rows:
        return pd.DataFrame()
    return _group_comparison_rows(comparison_rows)


def _build_algo3_comparison_frame(results_subdir: Path) -> pd.DataFrame:
    comparison_rows: list[dict[str, object]] = []
    if not results_subdir.is_dir():
        return pd.DataFrame()

    for model_dir in sorted(results_subdir.iterdir()):
        eval_dir = model_dir / "evaluated"
        if not eval_dir.is_dir():
            continue
        model = model_dir.name

        for eval_file in sorted(eval_dir.glob("*.csv")):
            evaluated = pd.read_csv(eval_file)

            for _, row in evaluated.iterrows():
                source_edges = _parse_algo3_edge_list(row.get("Source Graph"))
                target_edges = _parse_algo3_edge_list(row.get("Target Graph"))
                mother_edges = _parse_algo3_edge_list(row.get("Mother Graph"))
                llm_result_edges = set(_parse_algo3_edge_list(row.get("Results")))
                k = len(llm_result_edges)
                ground_truth = set(
                    find_valid_connections(mother_edges, source_edges, target_edges)
                )

                for baseline_strategy in _COMPARISON_BASELINE_STRATEGIES:
                    baseline_edges = _sample_baseline_edges(
                        baseline_strategy=baseline_strategy,
                        k=k,
                        mother_edges=mother_edges,
                        subgraph1_edges=source_edges,
                        subgraph2_edges=target_edges,
                    )
                    comparison_rows.extend(
                        _build_algo3_metric_rows(
                            model=model,
                            baseline_strategy=baseline_strategy,
                            source_file=eval_file.name,
                            k=k,
                            llm_recall=float(row.get("Recall", 0)),
                            llm_edges=llm_result_edges,
                            baseline_edges=baseline_edges,
                            ground_truth=ground_truth,
                            mother_edges=mother_edges,
                            source_edges=source_edges,
                            target_edges=target_edges,
                        )
                    )

    if not comparison_rows:
        return pd.DataFrame()
    return _group_comparison_rows(comparison_rows)
