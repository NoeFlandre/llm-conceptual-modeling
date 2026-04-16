import ast
import re
from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.common.types import Edge


def parse_edge_list(value: str | None) -> list[Edge]:
    if value is None:
        return []

    text = str(value).strip()
    if not text or text.lower() in {"empty", "nan"}:
        return []

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = None
    else:
        if isinstance(parsed, (list, set, tuple)):
            edges: list[Edge] = []
            for item in parsed:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    edges.append((str(item[0]).strip(), str(item[1]).strip()))
            if edges:
                return edges

    pairs = re.findall(r"\(([^()]+?,[^()]+?)\)", text)
    edges = []
    for pair in pairs:
        parts = pair.split(",", 1)
        if len(parts) != 2:
            continue
        left_raw, right_raw = parts[0].strip(), parts[1].strip()
        left = _clean_edge_part(left_raw)
        right = _clean_edge_part(right_raw)

        if left.startswith("(") and left.endswith(")"):
            try:
                nested = ast.literal_eval(left)
            except (ValueError, SyntaxError):
                pass
            else:
                if isinstance(nested, (list, tuple)) and nested:
                    left = str(nested[0]).strip()

        edges.append((left, right))

    return edges


def _clean_edge_part(value: str) -> str:
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _nodes_from_edges(edges: list[Edge]) -> set[str]:
    nodes: set[str] = set()
    for left, right in edges:
        nodes.add(str(left))
        nodes.add(str(right))
    return nodes


def _build_undirected_adjacency(edges: list[Edge]) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {}
    for left, right in edges:
        left_text = str(left).strip()
        right_text = str(right).strip()
        if not left_text or not right_text:
            continue
        adjacency.setdefault(left_text, set()).add(right_text)
        adjacency.setdefault(right_text, set()).add(left_text)
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


def compute_recall_for_row(
    source_edges: list[Edge],
    target_edges: list[Edge],
    mother_edges: list[Edge],
    result_edges: list[Edge],
) -> float:
    mother_adjacency = _build_undirected_adjacency(mother_edges)
    predicted_adjacency = _build_undirected_adjacency(source_edges)
    target_adjacency = _build_undirected_adjacency(target_edges)
    for left, neighbors in target_adjacency.items():
        predicted_adjacency.setdefault(left, set()).update(neighbors)
    result_adjacency = _build_undirected_adjacency(result_edges)
    for left, neighbors in result_adjacency.items():
        predicted_adjacency.setdefault(left, set()).update(neighbors)

    source_nodes = _nodes_from_edges(source_edges)
    target_nodes = _nodes_from_edges(target_edges)

    mother_components = _connected_components(mother_adjacency)

    actual_positive_pairs: list[tuple[str, str]] = []
    for source_node in source_nodes:
        if source_node not in mother_components:
            continue
        for target_node in target_nodes:
            if target_node not in mother_components:
                continue
            if mother_components[source_node] == mother_components[target_node]:
                actual_positive_pairs.append((source_node, target_node))

    if not actual_positive_pairs:
        return 0.0

    predicted_components = _connected_components(predicted_adjacency)

    true_positives = 0
    for source_node, target_node in actual_positive_pairs:
        if (
            source_node in predicted_components
            and target_node in predicted_components
            and predicted_components[source_node] == predicted_components[target_node]
        ):
            true_positives += 1

    return true_positives / len(actual_positive_pairs)


def evaluate_results_file(input_csv_path: str | Path, output_csv_path: str | Path) -> None:
    dataframe = pd.read_csv(input_csv_path)

    required_columns = {"Source Graph", "Target Graph", "Mother Graph", "Results", "Recall"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        msg = f"Missing required columns: {sorted(missing_columns)}"
        raise ValueError(msg)

    recalls: list[float] = []
    for _, row in dataframe.iterrows():
        source_edges = parse_edge_list(row.get("Source Graph"))
        target_edges = parse_edge_list(row.get("Target Graph"))
        mother_edges = parse_edge_list(row.get("Mother Graph"))
        result_edges = parse_edge_list(row.get("Results"))
        recall = compute_recall_for_row(source_edges, target_edges, mother_edges, result_edges)

        recalls.append(round(recall, 6))

    dataframe["Recall"] = recalls
    dataframe.to_csv(output_csv_path, index=False)
