from __future__ import annotations

import ast
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, cast

import pandas as pd

from llm_conceptual_modeling.algo3.evaluation import compute_recall_for_row
from llm_conceptual_modeling.common.baseline import (
    propose_random_k_edges,
    propose_strategy_cross_subgraph_edges,
)
from llm_conceptual_modeling.common.connection_eval import find_valid_connections
from llm_conceptual_modeling.common.literals import parse_python_literal
from llm_conceptual_modeling.common.types import PathLike

_RANDOM_SEED = 42
_COMPARISON_BASELINE_STRATEGIES = [
    "random-k",
    "wordnet-ontology-match",
    "edit-distance",
]
_ALGO12_METRICS = ["accuracy", "precision", "recall"]
_ALGO3_METRICS = ["accuracy", "precision", "recall"]


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


def _cross_subgraph_pair_count(
    subgraph1_edges: list[tuple[str, str]],
    subgraph2_edges: list[tuple[str, str]],
) -> int:
    subgraph1_nodes = {node for edge in subgraph1_edges for node in edge}
    subgraph2_nodes = {node for edge in subgraph2_edges for node in edge}
    return len(subgraph1_nodes) * len(subgraph2_nodes)


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


def _compute_baseline_counts(
    *,
    baseline_strategy: str,
    k: int,
    mother_edges: list[tuple[str, str]],
    subgraph1_edges: list[tuple[str, str]],
    subgraph2_edges: list[tuple[str, str]],
    ground_truth: set[tuple[str, str]],
) -> dict[str, int]:
    return dict(
        _compute_baseline_counts_cached(
            baseline_strategy=baseline_strategy,
            k=k,
            mother_edges=tuple(mother_edges),
            subgraph1_edges=tuple(subgraph1_edges),
            subgraph2_edges=tuple(subgraph2_edges),
            ground_truth=tuple(sorted(ground_truth)),
        )
    )


@lru_cache(maxsize=None)
def _compute_baseline_counts_cached(
    *,
    baseline_strategy: str,
    k: int,
    mother_edges: tuple[tuple[str, str], ...],
    subgraph1_edges: tuple[tuple[str, str], ...],
    subgraph2_edges: tuple[tuple[str, str], ...],
    ground_truth: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, int], ...]:
    baseline_edges = _sample_baseline_edges(
        baseline_strategy=baseline_strategy,
        k=k,
        mother_edges=list(mother_edges),
        subgraph1_edges=list(subgraph1_edges),
        subgraph2_edges=list(subgraph2_edges),
    )
    proposed_edges = [*subgraph1_edges, *subgraph2_edges, *sorted(baseline_edges)]
    generated_connections = find_valid_connections(
        proposed_edges,
        list(subgraph1_edges),
        list(subgraph2_edges),
    )
    ground_truth_edges = set(ground_truth)
    return {
        "tp": len(generated_connections & ground_truth_edges),
        "fp": len(generated_connections - ground_truth_edges),
        "fn": len(ground_truth_edges - generated_connections),
    }.items()


def _sample_baseline_edges(
    *,
    baseline_strategy: str,
    k: int,
    mother_edges: list[tuple[str, str]],
    subgraph1_edges: list[tuple[str, str]],
    subgraph2_edges: list[tuple[str, str]],
) -> set[tuple[str, str]]:
    mother_key = tuple(mother_edges)
    subgraph1_key = tuple(subgraph1_edges)
    subgraph2_key = tuple(subgraph2_edges)
    if k == 0:
        return set()
    ranked_edges = _ranked_baseline_edges(
        baseline_strategy=baseline_strategy,
        mother_edges=mother_key,
        subgraph1_edges=subgraph1_key,
        subgraph2_edges=subgraph2_key,
    )
    return set(ranked_edges[:k])


@lru_cache(maxsize=None)
def _ranked_baseline_edges(
    *,
    baseline_strategy: str,
    mother_edges: tuple[tuple[str, str], ...],
    subgraph1_edges: tuple[tuple[str, str], ...],
    subgraph2_edges: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    if baseline_strategy == "random-k":
        return tuple(propose_random_k_edges(mother_edges, len(mother_edges), seed=_RANDOM_SEED))
    subgraph1_nodes = {node for edge in subgraph1_edges for node in edge}
    subgraph2_nodes = {node for edge in subgraph2_edges for node in edge}
    max_pair_count = len(subgraph1_nodes) * len(subgraph2_nodes)
    return tuple(
        propose_strategy_cross_subgraph_edges(
            mother_edges,
            subgraph1_edges,
            subgraph2_edges,
            strategy=baseline_strategy,
            sample_count=max(max_pair_count, len(mother_edges)),
        )
    )


def _raw_from_evaluated(eval_file: Path, model_dir: Path) -> Path | None:
    eval_name = eval_file.name.replace("metrics_", "").replace(".csv", "")
    raw_dir = model_dir / "raw"
    if not raw_dir.is_dir():
        return None
    for raw_file in sorted(raw_dir.glob("*.csv")):
        if eval_name in raw_file.name:
            return raw_file
    return None


def _parse_edges(value: str | object) -> list[tuple[str, str]]:
    return list(_parse_edges_cached(str(value)))


@lru_cache(maxsize=None)
def _parse_edges_cached(text: str) -> tuple[tuple[str, str], ...]:
    if text == "None":
        return tuple()
    try:
        parsed = parse_python_literal(text)
        if isinstance(parsed, (list, set, tuple)):
            edges: list[tuple[str, str]] = []
            for item in parsed:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    edges.append((str(item[0]).strip(), str(item[1]).strip()))
            return tuple(edges)
    except Exception:
        return tuple()
    return tuple()


def _parse_algo3_edge_list(value: str | object) -> list[tuple[str, str]]:
    return list(_parse_algo3_edge_list_cached(str(value)))


@lru_cache(maxsize=None)
def _parse_algo3_edge_list_cached(text: str) -> tuple[tuple[str, str], ...]:
    if text == "None":
        return tuple()
    text = text.strip()
    if not text or text.lower() in {"empty", "nan"}:
        return tuple()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, set, tuple)):
            parsed_edges: list[tuple[str, str]] = []
            for item in parsed:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    parsed_edges.append((str(item[0]).strip(), str(item[1]).strip()))
            if parsed_edges:
                return tuple(parsed_edges)
    except Exception:
        pass

    pairs = re.findall(r"\(([^()]+?,[^()]+?)\)", text)
    edges: list[tuple[str, str]] = []
    for pair in pairs:
        parts = pair.split(",", 1)
        if len(parts) != 2:
            continue
        left = parts[0].strip().strip("'\"")
        right = parts[1].strip().strip("'\"")
        if left and right:
            edges.append((left, right))
    return tuple(edges)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


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
