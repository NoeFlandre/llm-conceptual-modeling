import networkx as nx
import pandas as pd

from llm_conceptual_modeling.algo1.evaluation import evaluate_results_file
from llm_conceptual_modeling.algo1.factorial import (
    run_factorial_analysis as run_algo1_factorial_analysis,
)
from llm_conceptual_modeling.algo2.evaluation import (
    evaluate_results_file as evaluate_algo2_results_file,
)
from llm_conceptual_modeling.algo2.factorial import (
    run_factorial_analysis as run_algo2_factorial_analysis,
)
from llm_conceptual_modeling.algo3.evaluation import (
    evaluate_results_file as evaluate_algo3_results_file,
)
from llm_conceptual_modeling.algo3.factorial import (
    run_factorial_analysis as run_algo3_factorial_analysis,
)
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


def test_algo2_evaluation_matches_legacy_metrics_fixture(tmp_path) -> None:
    raw_path = (
        "tests/fixtures/legacy/algo2/gpt-5/raw/algorithm2_results_sg1_sg2.csv"
    )
    expected_path = (
        "tests/fixtures/legacy/algo2/gpt-5/evaluated/metrics_sg1_sg2.csv"
    )
    output_path = tmp_path / "metrics_sg1_sg2.csv"

    evaluate_algo2_results_file(raw_path, output_path)

    actual = pd.read_csv(output_path)
    expected = pd.read_csv(expected_path)

    pd.testing.assert_series_equal(actual["accuracy"], expected["accuracy"], check_names=False)
    pd.testing.assert_series_equal(actual["recall"], expected["recall"], check_names=False)
    pd.testing.assert_series_equal(actual["precision"], expected["precision"], check_names=False)


def test_algo3_evaluation_matches_legacy_recall_fixture(tmp_path) -> None:
    raw_path = "tests/fixtures/legacy/algo3/gpt-5/raw/method3_results_gpt5.csv"
    expected_path = "tests/fixtures/legacy/algo3/gpt-5/evaluated/method3_results_evaluated_gpt5.csv"
    output_path = tmp_path / "method3_results_evaluated_gpt5.csv"

    evaluate_algo3_results_file(raw_path, output_path)

    actual = pd.read_csv(output_path)
    expected = pd.read_csv(expected_path)

    pd.testing.assert_series_equal(actual["Recall"], expected["Recall"], check_names=False)


def test_algo1_factorial_matches_legacy_fixture(tmp_path) -> None:
    input_paths = [
        "tests/fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg1_sg2.csv",
        "tests/fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg2_sg3.csv",
        "tests/fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg3_sg1.csv",
    ]
    expected_path = (
        "tests/fixtures/legacy/algo1/gpt-5/factorial/factorial_analysis_algo1_gpt_5_without_error.csv"
    )
    output_path = tmp_path / "factorial_analysis_algo1_gpt_5_without_error.csv"

    run_algo1_factorial_analysis(input_paths, output_path)

    actual = pd.read_csv(output_path)
    expected = pd.read_csv(expected_path)

    pd.testing.assert_frame_equal(actual, expected)


def test_algo2_factorial_matches_legacy_fixture(tmp_path) -> None:
    input_paths = [
        "tests/fixtures/legacy/algo2/gpt-5/evaluated/metrics_sg1_sg2.csv",
        "tests/fixtures/legacy/algo2/gpt-5/evaluated/metrics_sg2_sg3.csv",
        "tests/fixtures/legacy/algo2/gpt-5/evaluated/metrics_sg3_sg1.csv",
    ]
    expected_path = (
        "tests/fixtures/legacy/algo2/gpt-5/factorial/factorial_analysis_gpt_5_algo2_without_error.csv.csv"
    )
    output_path = tmp_path / "factorial_analysis_gpt_5_algo2_without_error.csv"

    run_algo2_factorial_analysis(input_paths, output_path)

    actual = pd.read_csv(output_path)
    expected = pd.read_csv(expected_path)

    pd.testing.assert_frame_equal(actual, expected)


def test_algo3_factorial_matches_legacy_fixture(tmp_path) -> None:
    input_path = "tests/fixtures/legacy/algo3/gpt-5/evaluated/method3_results_evaluated_gpt5.csv"
    expected_path = (
        "tests/fixtures/legacy/algo3/gpt-5/factorial/factorial_analysis_results_gpt5_without_error.csv"
    )
    output_path = tmp_path / "factorial_analysis_results_gpt5_without_error.csv"

    run_algo3_factorial_analysis(input_path, output_path)

    actual = pd.read_csv(output_path)
    expected = pd.read_csv(expected_path)

    pd.testing.assert_frame_equal(actual, expected)
