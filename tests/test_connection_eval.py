import networkx as nx
import pandas as pd

from llm_conceptual_modeling.algo1.evaluation import evaluate_results_file
from llm_conceptual_modeling.common.connection_eval import find_valid_connections


def test_find_valid_connections_blocks_paths_that_traverse_other_subgraph_nodes() -> None:
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            ("a1", "a2"),
            ("a2", "b1"),
            ("mid", "b2"),
        ]
    )
    subgraph1 = [("a1", "a2")]
    subgraph2 = [("b1", "b2")]

    assert find_valid_connections(graph, subgraph1, subgraph2) == {("a2", "b1")}


def test_algo1_evaluation_matches_legacy_metrics_fixture(tmp_path) -> None:
    raw_path = (
        "tests/fixtures/legacy/algo1/gpt-5/raw/algorithm1_results_sg1_sg2.csv"
    )
    expected_path = (
        "tests/fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg1_sg2.csv"
    )
    output_path = tmp_path / "metrics_sg1_sg2.csv"

    evaluate_results_file(raw_path, output_path)

    actual = pd.read_csv(output_path)
    expected = pd.read_csv(expected_path)

    pd.testing.assert_series_equal(actual["accuracy"], expected["accuracy"], check_names=False)
    pd.testing.assert_series_equal(actual["recall"], expected["recall"], check_names=False)
    pd.testing.assert_series_equal(actual["precision"], expected["precision"], check_names=False)
