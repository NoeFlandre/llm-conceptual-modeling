"""Metric-row construction helpers for baseline comparison.

Extracts row-building logic from baseline_bundle.py for algo1/2 and algo3
comparison frames.
"""

from __future__ import annotations

import pandas as pd

from llm_conceptual_modeling.algo3.evaluation import compute_recall_for_row

_ALGO12_METRICS = ["accuracy", "precision", "recall"]
_ALGO3_METRICS = ["accuracy", "precision", "recall"]


def _cross_subgraph_pair_count(
    subgraph1_edges: list[tuple[str, str]],
    subgraph2_edges: list[tuple[str, str]],
) -> int:
    subgraph1_nodes = {node for edge in subgraph1_edges for node in edge}
    subgraph2_nodes = {node for edge in subgraph2_edges for node in edge}
    return len(subgraph1_nodes) * len(subgraph2_nodes)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _group_comparison_rows(comparison_rows: list[dict[str, object]]) -> pd.DataFrame:
    comparison_frame = pd.DataFrame(comparison_rows)
    return (
        comparison_frame.groupby(
            ["algorithm", "model", "baseline_strategy", "metric"],
            dropna=False,
        )
        .agg(
            llm_mean=("llm_metric", "mean"),
            baseline_mean=("baseline_metric", "mean"),
            mean_delta=("delta", "mean"),
        )
        .reset_index()
    )


def _build_algo12_metric_rows(
    *,
    algo: str,
    model: str,
    baseline_strategy: str,
    source_file: str,
    llm_accuracy: float,
    llm_precision: float,
    llm_recall: float,
    k: int,
    baseline_tp: int,
    baseline_fp: int,
    baseline_fn: int,
    subgraph1_edges: list[tuple[str, str]],
    subgraph2_edges: list[tuple[str, str]],
) -> list[dict[str, object]]:
    baseline_tn = _cross_subgraph_pair_count(subgraph1_edges, subgraph2_edges) - (
        baseline_tp + baseline_fp + baseline_fn
    )

    rows: list[dict[str, object]] = []
    for metric in _ALGO12_METRICS:
        if metric == "accuracy":
            llm_metric = llm_accuracy
            baseline_metric = _safe_div(
                baseline_tp + baseline_tn,
                baseline_tp + baseline_fp + baseline_fn + baseline_tn,
            )
        elif metric == "precision":
            llm_metric = llm_precision
            baseline_metric = _safe_div(baseline_tp, baseline_tp + baseline_fp)
        else:
            llm_metric = llm_recall
            baseline_metric = _safe_div(baseline_tp, baseline_tp + baseline_fn)

        rows.append(
            {
                "algorithm": algo,
                "model": model,
                "baseline_strategy": baseline_strategy,
                "metric": metric,
                "source_file": source_file,
                "k": k,
                "llm_metric": llm_metric,
                "baseline_metric": baseline_metric,
                "delta": llm_metric - baseline_metric,
            }
        )
    return rows


def _build_algo3_metric_rows(
    *,
    model: str,
    baseline_strategy: str,
    source_file: str,
    k: int,
    llm_recall: float,
    llm_edges: set[tuple[str, str]],
    baseline_edges: set[tuple[str, str]],
    ground_truth: set[tuple[str, str]],
    mother_edges: list[tuple[str, str]],
    source_edges: list[tuple[str, str]],
    target_edges: list[tuple[str, str]],
) -> list[dict[str, object]]:
    baseline_tp = len(baseline_edges & ground_truth)
    baseline_fp = len(baseline_edges - ground_truth)
    baseline_fn = len(ground_truth - baseline_edges)
    baseline_tn = _cross_subgraph_pair_count(source_edges, target_edges) - (
        baseline_tp + baseline_fp + baseline_fn
    )

    llm_tp = len(llm_edges & ground_truth)
    llm_fp = len(llm_edges - ground_truth)
    llm_fn = len(ground_truth - llm_edges)
    llm_tn = _cross_subgraph_pair_count(source_edges, target_edges) - (
        llm_tp + llm_fp + llm_fn
    )
    baseline_recall = compute_recall_for_row(
        source_edges,
        target_edges,
        mother_edges,
        sorted(baseline_edges),
    )

    rows: list[dict[str, object]] = []
    for metric in _ALGO3_METRICS:
        if metric == "accuracy":
            llm_metric = _safe_div(
                llm_tp + llm_tn,
                llm_tp + llm_fp + llm_fn + llm_tn,
            )
            baseline_metric = _safe_div(
                baseline_tp + baseline_tn,
                baseline_tp + baseline_fp + baseline_fn + baseline_tn,
            )
        elif metric == "precision":
            llm_metric = _safe_div(llm_tp, llm_tp + llm_fp)
            baseline_metric = _safe_div(baseline_tp, baseline_tp + baseline_fp)
        else:
            llm_metric = llm_recall
            baseline_metric = baseline_recall

        rows.append(
            {
                "algorithm": "algo3",
                "model": model,
                "baseline_strategy": baseline_strategy,
                "metric": metric,
                "source_file": source_file,
                "k": k,
                "llm_metric": llm_metric,
                "baseline_metric": baseline_metric,
                "delta": llm_metric - baseline_metric,
            }
        )
    return rows
