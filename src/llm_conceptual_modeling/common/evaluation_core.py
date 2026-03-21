import networkx as nx
import pandas as pd

from llm_conceptual_modeling.common.connection_eval import find_valid_connections
from llm_conceptual_modeling.common.csv_schema import assert_output_columns, assert_required_columns
from llm_conceptual_modeling.common.literals import parse_python_literal
from llm_conceptual_modeling.common.types import PathLike


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


def evaluate_connection_results_file(
    input_csv_path: PathLike,
    output_csv_path: PathLike,
) -> None:
    dataframe = pd.read_csv(input_csv_path, delimiter=",")
    assert_required_columns(
        dataframe,
        {"graph", "subgraph1", "subgraph2", "Result"},
    )

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
        generated_connections = find_valid_connections(proposed_graph, subgraph_1, subgraph_2)

        nodes1 = {node for edge in subgraph_1 for node in edge}
        nodes2 = {node for edge in subgraph_2 for node in edge}
        tp, fp, fn, tn = _compute_confusion_counts(
            generated_connections,
            ground_truth_connections,
            nodes1,
            nodes2,
        )

        accuracy = 0.0
        if generated_connections:
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
    assert_output_columns(
        dataframe,
        list(dataframe.columns[:-3]) + ["accuracy", "recall", "precision"],
    )
    dataframe.to_csv(output_csv_path, index=False)
