import ast
import json
import re
from typing import Any

import pandas as pd

from llm_conceptual_modeling.algo3.evaluation import compute_recall_for_row, parse_edge_list
from llm_conceptual_modeling.common.connection_eval import find_valid_connections
from llm_conceptual_modeling.common.literals import parse_python_literal

Edge = tuple[str, str]


def extract_edge_list_from_chat_content(content: str) -> list[Edge]:
    stripped_content = content.strip()
    if stripped_content.startswith("{"):
        parsed_object = json.loads(stripped_content)
        return _extract_edges_from_json_object(parsed_object)

    list_match = re.search(r"\[[\s\S]*\]", stripped_content)
    if list_match is None:
        raise ValueError("Could not find a Python edge list in model content.")

    parsed_list = ast.literal_eval(list_match.group(0))
    return _normalize_edge_list(parsed_list)


def score_algo1_row(row: pd.Series, result_content: str) -> dict[str, float]:
    return score_connection_row(row, result_content)


def score_connection_row(row: pd.Series, result_content: str) -> dict[str, float]:
    edge_list = parse_python_literal(result_content)
    subgraph_1 = parse_python_literal(row["subgraph1"])
    subgraph_2 = parse_python_literal(row["subgraph2"])
    graph_edges = parse_python_literal(row["graph"])

    ground_truth_connections = find_valid_connections(graph_edges, subgraph_1, subgraph_2)

    proposed_edges = list(subgraph_1) + list(subgraph_2) + list(edge_list)
    generated_connections = find_valid_connections(proposed_edges, subgraph_1, subgraph_2)
    metrics = _compute_connection_metrics(
        generated_connections=generated_connections,
        ground_truth_connections=ground_truth_connections,
        subgraph_1=subgraph_1,
        subgraph_2=subgraph_2,
    )
    return metrics


def score_algo3_row(row: pd.Series, result_content: str) -> dict[str, float]:
    source_edges = parse_edge_list(row["Source Graph"])
    target_edges = parse_edge_list(row["Target Graph"])
    mother_edges = parse_edge_list(row["Mother Graph"])
    result_edges = parse_edge_list(result_content)
    recall = compute_recall_for_row(
        source_edges=source_edges,
        target_edges=target_edges,
        mother_edges=mother_edges,
        result_edges=result_edges,
    )
    return {"recall": recall}


def _extract_edges_from_json_object(parsed_object: dict[str, Any]) -> list[Edge]:
    edge_items = parsed_object["edges"]
    edges: list[Edge] = []
    for edge_item in edge_items:
        source = str(edge_item["source"])
        target = str(edge_item["target"])
        edges.append((source, target))
    return edges


def _normalize_edge_list(parsed_list: Any) -> list[Edge]:
    edges: list[Edge] = []
    for item in parsed_list:
        left_raw, right_raw = item
        left = str(left_raw)
        right = str(right_raw)
        edges.append((left, right))
    return edges


def _compute_connection_metrics(
    *,
    generated_connections: set[Edge],
    ground_truth_connections: set[Edge],
    subgraph_1: list[Edge],
    subgraph_2: list[Edge],
) -> dict[str, float]:
    true_positive_count = len(generated_connections & ground_truth_connections)
    false_positive_count = len(generated_connections - ground_truth_connections)
    false_negative_count = len(ground_truth_connections - generated_connections)
    node_count_left = len({node for edge in subgraph_1 for node in edge})
    node_count_right = len({node for edge in subgraph_2 for node in edge})
    true_negative_count = (node_count_left * node_count_right) - (
        true_positive_count + false_positive_count + false_negative_count
    )

    accuracy = 0.0
    has_generated_connections = bool(generated_connections)
    if has_generated_connections:
        total_count = (
            true_positive_count + false_positive_count + false_negative_count + true_negative_count
        )
        accuracy = (true_positive_count + true_negative_count) / total_count

    recall_denominator = true_positive_count + false_negative_count
    recall = true_positive_count / recall_denominator if recall_denominator else 0.0
    precision_denominator = true_positive_count + false_positive_count
    precision = true_positive_count / precision_denominator if precision_denominator else 0.0
    return {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
    }
