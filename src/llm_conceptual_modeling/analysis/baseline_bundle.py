from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.common.baseline import propose_random_k_edges
from llm_conceptual_modeling.common.connection_eval import find_valid_connections
from llm_conceptual_modeling.common.literals import parse_python_literal
from llm_conceptual_modeling.common.types import PathLike

# Fixed seed for reproducibility across runs and environments
_RANDOM_SEED = 42


def write_baseline_comparison_bundle(
    *,
    results_root: PathLike,
    output_dir: PathLike,
) -> None:
    """Generate the organized baseline-comparison bundle using the random-k strategy.

    For each LLM output row, the baseline samples exactly k edges (where k equals
    the number of edges the LLM proposed) from the mother graph, with no knowledge
    of which edges are cross-subgraph.  Both the LLM and the baseline are evaluated
    against the ground-truth cross edges, giving a fair, volume-matched comparison.

    Produces:
        <output_dir>/
            README.md
            bundle_manifest.csv
            baseline_advantage_summary.csv
            algo1_model_vs_baseline.csv
            algo2_model_vs_baseline.csv
            algo3_model_vs_baseline.csv
            all_models_vs_baseline.csv
    """
    results_root_path = Path(results_root)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    manifest_records: list[dict[str, object]] = []

    _run_algo_comparison(
        algo="algo1",
        results_subdir=results_root_path / "algo1",
        output_dir=output_dir_path,
        metrics=["accuracy", "precision", "recall"],
        manifest_records=manifest_records,
    )
    _run_algo_comparison(
        algo="algo2",
        results_subdir=results_root_path / "algo2",
        output_dir=output_dir_path,
        metrics=["accuracy", "precision", "recall"],
        manifest_records=manifest_records,
    )
    _run_algo3_comparison(
        results_subdir=results_root_path / "algo3",
        output_dir=output_dir_path,
        manifest_records=manifest_records,
    )

    # --- Build summary table ---
    summary_path = output_dir_path / "baseline_advantage_summary.csv"
    _write_advantage_summary(output_dir_path, summary_path)
    manifest_records.append(
        {
            "file": str(summary_path.name),
            "description": (
                "Cross-model summary: for each algorithm and metric, how many models "
                "beat the random-k baseline and by how much. "
                "Columns: algorithm, metric, model_count, models_beating_baseline, "
                "best_model, best_model_delta, worst_model, worst_model_delta, "
                "average_model_delta."
            ),
        }
    )

    # --- Write bundle metadata ---
    pd.DataFrame.from_records(manifest_records).to_csv(
        output_dir_path / "bundle_manifest.csv",
        index=False,
    )

    _write_bundle_readme(output_dir_path)


# ---------------------------------------------------------------------------
# ALGO1 / ALGO2 per-row comparison
# ---------------------------------------------------------------------------

_METRICS_BY_ALGO: dict[str, list[str]] = {
    "algo1": ["accuracy", "precision", "recall"],
    "algo2": ["accuracy", "precision", "recall"],
}


def _run_algo_comparison(
    *,
    algo: str,
    results_subdir: Path,
    output_dir: Path,
    metrics: list[str],
    manifest_records: list[dict[str, object]],
) -> None:
    """Compare ALGO1/ALGO2 LLM outputs against random-k baseline per-row."""
    comparison_rows: list[dict[str, object]] = []

    if not results_subdir.is_dir():
        return

    model_dirs = sorted(results_subdir.iterdir())
    for model_dir in model_dirs:
        eval_dir = model_dir / "evaluated"
        if not eval_dir.is_dir():
            continue
        model = model_dir.name

        for eval_file in sorted(eval_dir.glob("metrics_*.csv")):
            evaluated = pd.read_csv(eval_file)
            # Read the raw file to get the mother's Result edges
            raw_file = _raw_from_evaluated(eval_file, model_dir)
            if raw_file is None:
                continue
            raw = pd.read_csv(raw_file)

            for idx, row in evaluated.iterrows():
                raw_row = raw.iloc[idx]
                llm_result_edges = _parse_edges(raw_row.get("Result", "[]"))
                k = len(llm_result_edges)

                mother_edges = _parse_edges(raw_row.get("graph", "[]"))
                sg1_edges = _parse_edges(raw_row.get("subgraph1", "[]"))
                sg2_edges = _parse_edges(raw_row.get("subgraph2", "[]"))

                # Ground-truth cross edges
                ground_truth = find_valid_connections(mother_edges, sg1_edges, sg2_edges)

                # Candidate set = mother graph (no structure knowledge)
                candidate_edges = list(mother_edges)

                # Baseline: sample k from candidate
                if k == 0:
                    baseline_tp = 0
                    baseline_fp = 0
                    baseline_fn = len(ground_truth)
                else:
                    baseline_sampled = set(
                        propose_random_k_edges(candidate_edges, k, seed=_RANDOM_SEED)
                    )
                    baseline_tp = len(baseline_sampled & ground_truth)
                    baseline_fp = len(baseline_sampled - ground_truth)
                    baseline_fn = len(ground_truth - baseline_sampled)

                # LLM: compute from row metrics
                sg1_nodes = {n for e in sg1_edges for n in e}
                sg2_nodes = {n for e in sg2_edges for n in e}
                tn = len(sg1_nodes) * len(sg2_nodes) - (
                    baseline_tp + baseline_fp + baseline_fn
                )

                llm_tp = int(round(row.get("precision", 0) * k)) if k > 0 else 0
                llm_fp = k - llm_tp
                llm_fn = baseline_fn  # same ground truth
                llm_tn = tn
                baseline_tn = tn  # same TN for same ground truth

                for metric in metrics:
                    if metric == "accuracy":
                        llm_val = _safe_div(llm_tp + llm_tn, llm_tp + llm_fp + llm_fn + llm_tn)
                        baseline_val = _safe_div(
                            baseline_tp + baseline_tn,
                            baseline_tp + baseline_fp + baseline_fn + baseline_tn,
                        )
                    elif metric == "precision":
                        llm_val = _safe_div(llm_tp, llm_tp + llm_fp)
                        baseline_val = _safe_div(baseline_tp, baseline_tp + baseline_fp)
                    else:  # recall
                        llm_val = _safe_div(llm_tp, llm_tp + llm_fn)
                        baseline_val = _safe_div(baseline_tp, baseline_tp + baseline_fn)

                    comparison_rows.append(
                        {
                            "algorithm": algo,
                            "model": model,
                            "metric": metric,
                            "source_file": eval_file.name,
                            "k": k,
                            "llm_metric": llm_val,
                            "baseline_metric": baseline_val,
                            "delta": llm_val - baseline_val,
                        }
                    )

    if not comparison_rows:
        return

    comp_df = pd.DataFrame(comparison_rows)

    # Per-model grouped summary
    grouped = (
        comp_df.groupby(["algorithm", "model", "metric"], dropna=False)
        .agg(
            llm_mean=("llm_metric", "mean"),
            baseline_mean=("baseline_metric", "mean"),
            mean_delta=("delta", "mean"),
        )
        .reset_index()
    )
    out_path = output_dir / f"{algo}_model_vs_baseline.csv"
    grouped.to_csv(out_path, index=False)
    manifest_records.append(
        {
            "file": out_path.name,
            "description": (
                f"{algo.upper()} per-model comparison against the random-k baseline "
                "(samples k edges from mother graph, k = LLM's edge count per row). "
                "Columns: algorithm, model, metric, llm_mean, baseline_mean, mean_delta."
            ),
        }
    )

    # All-model combined
    all_out = output_dir / "all_models_vs_baseline.csv"
    all_combined = (
        pd.concat([grouped], ignore_index=True)
        if (all_out.exists() and False)
        else grouped
    )
    if all_out.exists():
        existing = pd.read_csv(all_out)
        all_combined = pd.concat([existing, grouped], ignore_index=True)
    else:
        all_combined = grouped
    all_combined.to_csv(all_out, index=False)


def _raw_from_evaluated(eval_file: Path, model_dir: Path) -> Path | None:
    """Find the raw CSV that produced an evaluated CSV."""
    eval_name = eval_file.name.replace("metrics_", "").replace(".csv", "")
    # Look in the raw directory
    raw_dir = model_dir / "raw"
    if not raw_dir.is_dir():
        return None
    for raw_file in sorted(raw_dir.glob("*.csv")):
        if eval_name in raw_file.name:
            return raw_file
    return None


# ---------------------------------------------------------------------------
# ALGO3 per-row comparison
# ---------------------------------------------------------------------------

_ALGO3_METRICS = ["recall"]


def _parse_edges(value: str | object) -> list[tuple[str, str]]:
    """Parse an edge list from a string representation."""
    if value is None:
        return []
    try:
        parsed = parse_python_literal(str(value))
        if isinstance(parsed, (list, set, tuple)):
            edges = []
            for item in parsed:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    edges.append((str(item[0]).strip(), str(item[1]).strip()))
            return edges
    except Exception:
        pass
    return []


def _build_undirected_adjacency(
    edges: list[tuple[str, str]],
) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {}
    for left, right in edges:
        if not left or not right:
            continue
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    return adjacency


def _connected_components(adjacency: dict[str, set[str]]) -> dict[str, int]:
    components: dict[str, int] = {}
    component_index = 0
    for start_node in adjacency:
        if start_node in components:
            continue
        stack = [start_node]
        components[start_node] = component_index
        while stack:
            current = stack.pop()
            for neighbor in adjacency.get(current, set()):
                if neighbor in components:
                    continue
                components[neighbor] = component_index
                stack.append(neighbor)
        component_index += 1
    return components


def _compute_connectivity_recall(
    sampled_edges: list[tuple[str, str]],
    source_edges: list[tuple[str, str]],
    target_edges: list[tuple[str, str]],
    mother_edges: list[tuple[str, str]],
) -> float:
    """Same recall metric as ALGO3 evaluation: connectivity-based."""
    predicted_adj = _build_undirected_adjacency(sampled_edges)
    for left, right in source_edges + target_edges + sampled_edges:
        predicted_adj.setdefault(left, set()).add(right)
        predicted_adj.setdefault(right, set()).add(left)

    source_nodes = {n for e in source_edges for n in e}
    target_nodes = {n for e in target_edges for n in e}

    mother_adj = _build_undirected_adjacency(mother_edges)
    mother_components = _connected_components(mother_adj)

    true_cross_pairs = []
    for s in source_nodes:
        if s not in mother_components:
            continue
        for t in target_nodes:
            if t not in mother_components:
                continue
            if mother_components[s] == mother_components[t]:
                true_cross_pairs.append((s, t))

    if not true_cross_pairs:
        return 0.0

    pred_components = _connected_components(predicted_adj)
    true_positives = sum(
        1
        for s, t in true_cross_pairs
        if s in pred_components
        and t in pred_components
        and pred_components[s] == pred_components[t]
    )
    return true_positives / len(true_cross_pairs)


def _parse_algo3_edge_list(value: str | object) -> list[tuple[str, str]]:
    """Parse ALGO3 edge list from string, handling its format."""
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() in {"empty", "nan"}:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, set, tuple)):
            edges = []
            for item in parsed:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    edges.append((str(item[0]).strip(), str(item[1]).strip()))
            if edges:
                return edges
    except Exception:
        pass
    pairs = re.findall(r"\(([^()]+?,[^()]+?)\)", text)
    edges = []
    for pair in pairs:
        parts = pair.split(",", 1)
        if len(parts) != 2:
            continue
        left = parts[0].strip().strip("'\"")
        right = parts[1].strip().strip("'\"")
        if left and right:
            edges.append((left, right))
    return edges


def _run_algo3_comparison(
    *,
    results_subdir: Path,
    output_dir: Path,
    manifest_records: list[dict[str, object]],
) -> None:
    """Compare ALGO3 LLM outputs against random-k baseline per-row.

    Uses the same edge-overlap metrics (precision, recall) as ALGO1/2.
    The baseline samples k edges from the mother graph (not source×target pairs),
    matching the ALGO1/2 strategy.  Note: ALGO3's connectivity-based Recall metric
    from the original evaluation is not directly comparable using this approach;
    here we report standard edge-overlap precision and recall.
    """
    comparison_rows: list[dict[str, object]] = []

    if not results_subdir.is_dir():
        return

    model_dirs = sorted(results_subdir.iterdir())
    for model_dir in model_dirs:
        eval_dir = model_dir / "evaluated"
        if not eval_dir.is_dir():
            continue
        model = model_dir.name

        for eval_file in sorted(eval_dir.glob("*.csv")):
            evaluated = pd.read_csv(eval_file)

            for _idx, row in evaluated.iterrows():
                source_edges = _parse_algo3_edge_list(row.get("Source Graph"))
                target_edges = _parse_algo3_edge_list(row.get("Target Graph"))
                mother_edges = _parse_algo3_edge_list(row.get("Mother Graph"))
                llm_result_edges = _parse_algo3_edge_list(row.get("Results"))
                k = len(llm_result_edges)

                # Ground-truth cross edges from mother graph
                ground_truth = set(
                    find_valid_connections(mother_edges, source_edges, target_edges)
                )

                # Baseline: sample k edges uniformly from mother graph (same strategy as ALGO1/2)
                candidate_edges = list(mother_edges)
                if k == 0:
                    sampled_set: set[tuple[str, str]] = set()
                else:
                    sampled_set = set(propose_random_k_edges(candidate_edges, k, seed=_RANDOM_SEED))

                baseline_tp = len(sampled_set & ground_truth)
                baseline_fp = len(sampled_set - ground_truth)
                baseline_fn = len(ground_truth - sampled_set)

                # LLM metrics
                llm_edges = set(llm_result_edges)
                llm_tp = len(llm_edges & ground_truth)
                llm_fp = len(llm_edges - ground_truth)
                llm_fn = baseline_fn  # same ground truth

                for metric in ["recall", "precision", "accuracy"]:
                    if metric == "recall":
                        llm_val = _safe_div(llm_tp, llm_tp + llm_fn)
                        baseline_val = _safe_div(baseline_tp, baseline_tp + baseline_fn)
                    elif metric == "precision":
                        llm_val = _safe_div(llm_tp, llm_tp + llm_fp)
                        baseline_val = _safe_div(baseline_tp, baseline_tp + baseline_fp)
                    else:  # accuracy
                        sg1_nodes = {n for e in source_edges for n in e}
                        sg2_nodes = {n for e in target_edges for n in e}
                        tn = len(sg1_nodes) * len(sg2_nodes) - (
                            baseline_tp + baseline_fp + baseline_fn
                        )
                        total = llm_tp + llm_fp + llm_fn + tn
                        llm_val = _safe_div(llm_tp + tn, total)
                        baseline_val = _safe_div(
                            baseline_tp + tn, baseline_tp + baseline_fp + baseline_fn + tn
                        )

                    comparison_rows.append(
                        {
                            "algorithm": "algo3",
                            "model": model,
                            "metric": metric,
                            "source_file": eval_file.name,
                            "k": k,
                            "llm_metric": llm_val,
                            "baseline_metric": baseline_val,
                            "delta": llm_val - baseline_val,
                        }
                    )

    if not comparison_rows:
        return

    comp_df = pd.DataFrame(comparison_rows)

    grouped = (
        comp_df.groupby(["algorithm", "model", "metric"], dropna=False)
        .agg(
            llm_mean=("llm_metric", "mean"),
            baseline_mean=("baseline_metric", "mean"),
            mean_delta=("delta", "mean"),
        )
        .reset_index()
    )
    out_path = output_dir / "algo3_model_vs_baseline.csv"
    grouped.to_csv(out_path, index=False)
    manifest_records.append(
        {
            "file": out_path.name,
            "description": (
                "ALGO3 per-model comparison against the random-k baseline "
                "(samples k pairs from source×target nodes, k = LLM's edge count per row). "
                "Columns: algorithm, model, metric, llm_mean, baseline_mean, mean_delta."
            ),
        }
    )

    # Append to all-model combined
    all_out = output_dir / "all_models_vs_baseline.csv"
    if all_out.exists():
        existing = pd.read_csv(all_out)
        all_combined = pd.concat([existing, grouped], ignore_index=True)
    else:
        all_combined = grouped
    all_combined.to_csv(all_out, index=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _write_advantage_summary(output_dir: Path, summary_path: Path) -> None:
    records: list[dict[str, object]] = []
    for algo in ["algo1", "algo2", "algo3"]:
        comp_file = output_dir / f"{algo}_model_vs_baseline.csv"
        if not comp_file.exists():
            continue
        df = pd.read_csv(comp_file)
        for metric in df["metric"].unique():
            metric_df = df[df["metric"] == metric]
            if len(metric_df) == 0:
                continue
            beating = metric_df[metric_df["mean_delta"] > 0]
            best_row = (
                beating.sort_values("mean_delta", ascending=False).iloc[0]
                if len(beating) > 0
                else metric_df.sort_values("mean_delta", ascending=False).iloc[0]
            )
            worst_row = metric_df.sort_values("mean_delta").iloc[0]
            records.append(
                {
                    "algorithm": algo,
                    "metric": metric,
                    "model_count": int(len(metric_df)),
                    "models_beating_baseline": int(len(beating)),
                    "best_model": best_row["model"],
                    "best_model_delta": float(best_row["mean_delta"]),
                    "worst_model": worst_row["model"],
                    "worst_model_delta": float(worst_row["mean_delta"]),
                    "average_model_delta": float(metric_df["mean_delta"].mean()),
                }
            )
    pd.DataFrame.from_records(records).to_csv(summary_path, index=False)


def _write_bundle_readme(output_dir: Path) -> None:
    readme = """# Non-LLM Baseline Comparison Bundle

This directory contains the organized artifacts for the non-LLM baseline comparison revision item.

## Purpose

The reviewer asked for a non-LLM comparator to contextualize the value proposition of using
LLMs despite their inherent variability. The baseline is the **random-k** strategy:
for each LLM output row, the baseline samples exactly k edges uniformly at random from the
mother graph, where k equals the number of edges the LLM proposed in that row.  No information
about which edges are cross-subgraph is used.

This gives a fair, volume-matched comparison: both the LLM and the baseline are allowed
the same number of guesses, and both are evaluated against the ground-truth cross edges.

## Baseline Strategy

**random-k** (seed=42):

1. For each LLM output row, extract k = number of edges the LLM proposed.
2. Sample exactly k edges uniformly at random from the mother graph.
3. Evaluate the sampled edges against the ground-truth cross edges (same as the LLM).
4. Compare: LLM metric vs. baseline metric for the same row, then average across rows.

This answers: does the LLM find true cross edges better than random guessing,
when both are allowed the same number of guesses?

## Layout

- `bundle_manifest.csv`
  Index of every generated file with descriptions.
- `baseline_advantage_summary.csv`
  Cross-model summary: for each algorithm and metric, how many models beat the baseline.
- `<algo>_model_vs_baseline.csv`
  Per-model comparison: mean LLM metric vs mean baseline metric, with delta.
- `all_models_vs_baseline.csv`
  Combined all-model comparison across all three algorithms.

## Interpretation

A positive `mean_delta` means the LLM outperforms random guessing on that metric.
A negative `mean_delta` means random guessing is more effective.
The comparison is fair: both methods propose the same number of edges per row.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
