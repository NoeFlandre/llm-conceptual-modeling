from pathlib import Path

import networkx as nx
import pandas as pd

from llm_conceptual_modeling.common.connection_eval import find_valid_connections
from llm_conceptual_modeling.common.literals import parse_python_literal


def _compute_confusion_counts(
    generated_connections: set[tuple[str, str]],
    ground_truth_connections: set[tuple[str, str]],
    nodes1: set[str],
    nodes2: set[str],
) -> tuple[int, int, int, int]:
    true_positive_count = len(generated_connections & ground_truth_connections)
    false_positive_count = len(generated_connections - ground_truth_connections)
    false_negative_count = len(ground_truth_connections - generated_connections)
    true_negative_count = (len(nodes1) * len(nodes2)) - (
        true_positive_count + false_positive_count + false_negative_count
    )
    return (
        true_positive_count,
        false_positive_count,
        false_negative_count,
        true_negative_count,
    )


def evaluate_results_file(input_csv_path: str | Path, output_csv_path: str | Path) -> None:
    dataframe = pd.read_csv(input_csv_path, delimiter=",")

    required_columns = {"graph", "subgraph1", "subgraph2", "Result"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        msg = f"Missing required columns: {sorted(missing_columns)}"
        raise ValueError(msg)

    dataframe["graph"] = dataframe["graph"].apply(parse_python_literal)
    dataframe["subgraph1"] = dataframe["subgraph1"].apply(parse_python_literal)
    dataframe["subgraph2"] = dataframe["subgraph2"].apply(parse_python_literal)

    accuracies: list[float] = []
    recalls: list[float] = []
    precisions: list[float] = []

    for _, row in dataframe.iterrows():
        edge_list = parse_python_literal(row["Result"])
        subgraph_1 = parse_python_literal(row["subgraph1"])
        subgraph_2 = parse_python_literal(row["subgraph2"])
        graph_edges = parse_python_literal(row["graph"])

        ground_truth_graph = nx.DiGraph()
        ground_truth_graph.add_edges_from(graph_edges)
        ground_truth_connections = find_valid_connections(
            ground_truth_graph,
            subgraph_1,
            subgraph_2,
        )

        proposed_graph = nx.DiGraph()
        proposed_graph.add_edges_from(subgraph_1)
        proposed_graph.add_edges_from(subgraph_2)
        proposed_graph.add_edges_from(edge_list)

        generated_connections = find_valid_connections(
            proposed_graph,
            subgraph_1,
            subgraph_2,
        )

        nodes1 = {node for edge in subgraph_1 for node in edge}
        nodes2 = {node for edge in subgraph_2 for node in edge}
        tp, fp, fn, tn = _compute_confusion_counts(
            generated_connections,
            ground_truth_connections,
            nodes1,
            nodes2,
        )

        if not generated_connections:
            accuracy = 0.0
        else:
            total = tp + fp + fn + tn
            accuracy = (tp + tn) / total if total > 0 else 0.0

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        accuracies.append(accuracy)
        recalls.append(recall)
        precisions.append(precision)

    dataframe["accuracy"] = accuracies
    dataframe["recall"] = recalls
    dataframe["precision"] = precisions
    dataframe.to_csv(output_csv_path, index=False)
