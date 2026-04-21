from __future__ import annotations

import re

from llm_conceptual_modeling.common.connection_eval import find_valid_connections
from llm_conceptual_modeling.common.literals import parse_python_literal


def connection_metric_summary(raw_row: dict[str, object]) -> dict[str, float]:
    graph = parse_python_literal(str(raw_row["graph"]))
    subgraph1 = parse_python_literal(str(raw_row["subgraph1"]))
    subgraph2 = parse_python_literal(str(raw_row["subgraph2"]))
    result_edges = parse_python_literal(str(raw_row["Result"]))
    ground_truth_connections = find_valid_connections(graph, subgraph1, subgraph2)
    proposed_edges = list(subgraph1) + list(subgraph2) + list(result_edges)
    generated_connections = find_valid_connections(proposed_edges, subgraph1, subgraph2)
    nodes1 = {node for edge in subgraph1 for node in edge}
    nodes2 = {node for edge in subgraph2 for node in edge}
    tp = len(generated_connections & ground_truth_connections)
    fp = len(generated_connections - ground_truth_connections)
    fn = len(ground_truth_connections - generated_connections)
    tn = (len(nodes1) * len(nodes2)) - (tp + fp + fn)
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    denominator = precision + recall
    f1 = (2 * precision * recall) / denominator if denominator > 0 else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def trace_metric_summary(records: list[dict[str, object]]) -> dict[str, float | int]:
    durations: list[float] = []
    tokens_per_second: list[float] = []
    completion_tokens: list[int] = []
    prompt_tokens: list[int] = []
    for record in records:
        metrics = _trace_metrics(record)
        if metrics is None:
            continue
        duration = metrics.get("duration_seconds")
        if isinstance(duration, (int, float)):
            durations.append(float(duration))
        tps = metrics.get("tokens_per_second")
        if isinstance(tps, (int, float)):
            tokens_per_second.append(float(tps))
        completion = metrics.get("completion_token_count")
        if isinstance(completion, int):
            completion_tokens.append(completion)
        prompt = metrics.get("prompt_token_count")
        if isinstance(prompt, int):
            prompt_tokens.append(prompt)
    if not durations and not tokens_per_second and not completion_tokens and not prompt_tokens:
        return {}
    summary: dict[str, float | int] = {"response_trace_count": len(records)}
    if durations:
        summary["avg_duration_seconds"] = sum(durations) / len(durations)
        summary["max_duration_seconds"] = max(durations)
    if tokens_per_second:
        summary["avg_tokens_per_second"] = sum(tokens_per_second) / len(tokens_per_second)
        summary["max_tokens_per_second"] = max(tokens_per_second)
    if completion_tokens:
        summary["total_completion_tokens"] = sum(completion_tokens)
    if prompt_tokens:
        summary["total_prompt_tokens"] = sum(prompt_tokens)
    return summary


def _trace_metrics(record: dict[str, object]) -> dict[str, object] | None:
    raw_metrics = record.get("metrics")
    if not isinstance(raw_metrics, dict):
        return None
    return {str(key): value for key, value in raw_metrics.items()}


def summary_from_raw_row(algorithm: str, raw_row: dict[str, object]) -> dict[str, object]:
    if algorithm in {"algo1", "algo2"} and "Result" in raw_row:
        result_edges = parse_python_literal(str(raw_row["Result"]))
        result_count = len(result_edges)
        return {
            "candidate_edge_count": result_count,
            "verified_edge_count": result_count,
            **connection_metric_summary(raw_row),
        }
    if algorithm == "algo3" and "Results" in raw_row:
        result_edges = parse_python_literal(str(raw_row["Results"]))
        return {
            "result_edge_count": len(result_edges),
            "recall": float(str(raw_row.get("Recall", 0.0))),
        }
    return {}


def validate_structural_runtime_result(*, algorithm: str, raw_row: dict[str, object]) -> None:
    if algorithm != "algo1":
        return
    result_literal = raw_row.get("Result")
    if result_literal is None:
        raise ValueError(f"Structurally invalid {algorithm} result: missing Result field.")
    result_edges = parse_python_literal(str(result_literal))
    sanitized_edges = sanitize_algorithm_edge_result(algorithm, result_edges)
    raw_row["Result"] = repr(sanitized_edges)


def sanitize_algorithm_edge_result(
    algorithm: str,
    result_edges: object,
) -> list[tuple[str, str]]:
    if not isinstance(result_edges, (list, tuple)):
        raise ValueError(f"Structurally invalid {algorithm} result: Result is not a list.")
    sanitized_edges: list[tuple[str, str]] = []
    for edge in result_edges:
        if not isinstance(edge, (list, tuple)) or len(edge) < 2:
            continue
        source = _normalize_structural_endpoint(edge[0], algorithm=algorithm)
        target = _normalize_structural_endpoint(edge[1], algorithm=algorithm)
        if source is None or target is None:
            continue
        if not _looks_like_textual_concept(source) or not _looks_like_textual_concept(target):
            continue
        sanitized_edges.append((source, target))
    return sanitized_edges


def _normalize_structural_endpoint(value: object, *, algorithm: str) -> str | None:
    text = str(value).strip()
    if not text:
        return None
    _ = algorithm
    return text


def _looks_like_textual_concept(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text))
