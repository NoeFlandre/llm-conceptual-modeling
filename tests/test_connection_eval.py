import networkx as nx
import pandas as pd
import pytest

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
from llm_conceptual_modeling.common.evaluation_core import evaluate_connection_results_file


def test_find_valid_connections_accepts_any_path_between_subgraphs() -> None:
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

    assert find_valid_connections(graph, subgraph1, subgraph2) == {
        ("a1", "b1"),
        ("a2", "b1"),
    }


def test_algo1_evaluation_matches_legacy_metrics_fixture(tmp_path) -> None:
    raw_path = "tests/fixtures/legacy/algo1/gpt-5/raw/algorithm1_results_sg1_sg2.csv"
    expected_path = "tests/fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg1_sg2.csv"
    output_path = tmp_path / "metrics_sg1_sg2.csv"

    evaluate_results_file(raw_path, output_path)

    actual = pd.read_csv(output_path)
    expected = pd.read_csv(expected_path)

    pd.testing.assert_series_equal(actual["accuracy"], expected["accuracy"], check_names=False)
    pd.testing.assert_series_equal(actual["recall"], expected["recall"], check_names=False)
    pd.testing.assert_series_equal(actual["precision"], expected["precision"], check_names=False)
    expected_f1 = (2 * expected["precision"] * expected["recall"]) / (
        expected["precision"] + expected["recall"]
    )
    expected_f1 = expected_f1.fillna(0.0)
    pd.testing.assert_series_equal(actual["f1"], expected_f1, check_names=False)


def test_algo2_evaluation_matches_legacy_metrics_fixture(tmp_path) -> None:
    raw_path = "tests/fixtures/legacy/algo2/gpt-5/raw/algorithm2_results_sg1_sg2.csv"
    expected_path = "tests/fixtures/legacy/algo2/gpt-5/evaluated/metrics_sg1_sg2.csv"
    output_path = tmp_path / "metrics_sg1_sg2.csv"

    evaluate_algo2_results_file(raw_path, output_path)

    actual = pd.read_csv(output_path)
    expected = pd.read_csv(expected_path)

    pd.testing.assert_series_equal(actual["accuracy"], expected["accuracy"], check_names=False)
    pd.testing.assert_series_equal(actual["recall"], expected["recall"], check_names=False)
    pd.testing.assert_series_equal(actual["precision"], expected["precision"], check_names=False)
    expected_f1 = (2 * expected["precision"] * expected["recall"]) / (
        expected["precision"] + expected["recall"]
    )
    expected_f1 = expected_f1.fillna(0.0)
    pd.testing.assert_series_equal(actual["f1"], expected_f1, check_names=False)


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
    expected_path = "tests/fixtures/legacy/algo1/gpt-5/factorial/factorial_analysis_algo1_gpt_5_without_error.csv"
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
    expected_path = "tests/fixtures/legacy/algo2/gpt-5/factorial/factorial_analysis_gpt_5_algo2_without_error.csv"
    output_path = tmp_path / "factorial_analysis_gpt_5_algo2_without_error.csv"

    run_algo2_factorial_analysis(input_paths, output_path)

    actual = pd.read_csv(output_path)
    expected = pd.read_csv(expected_path)

    pd.testing.assert_frame_equal(actual, expected)


def test_algo3_factorial_matches_legacy_fixture(tmp_path) -> None:
    input_path = "tests/fixtures/legacy/algo3/gpt-5/evaluated/method3_results_evaluated_gpt5.csv"
    expected_path = "tests/fixtures/legacy/algo3/gpt-5/factorial/factorial_analysis_results_gpt5_without_error.csv"
    output_path = tmp_path / "factorial_analysis_results_gpt5_without_error.csv"

    run_algo3_factorial_analysis(input_path, output_path)

    actual = pd.read_csv(output_path)
    expected = pd.read_csv(expected_path)

    pd.testing.assert_frame_equal(actual, expected)


def test_connection_accuracy_counts_true_negatives_for_empty_prediction(tmp_path) -> None:
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    pd.DataFrame(
        [
            {
                "graph": "[]",
                "subgraph1": "[('a', 'b')]",
                "subgraph2": "[('c', 'd')]",
                "Result": "[]",
            }
        ]
    ).to_csv(input_path, index=False)

    evaluate_connection_results_file(input_path, output_path)

    actual = pd.read_csv(output_path)
    assert actual.loc[0, "accuracy"] == 1.0
    assert actual.loc[0, "recall"] == 0.0
    assert actual.loc[0, "precision"] == 0.0


def test_algo3_evaluation_raises_on_bad_recall_input(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "bad_method3.csv"
    output_path = tmp_path / "bad_method3_out.csv"
    pd.DataFrame(
        [
            {
                "Source Graph": "[('a', 'b')]",
                "Target Graph": "[('c', 'd')]",
                "Mother Graph": "[('a', 'c')]",
                "Results": "[('x', 'y')]",
                "Recall": 0.25,
            }
        ]
    ).to_csv(input_path, index=False)

    def boom(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("bad parse")

    monkeypatch.setattr(
        "llm_conceptual_modeling.algo3.evaluation.compute_recall_for_row",
        boom,
    )

    with pytest.raises(RuntimeError, match="bad parse"):
        evaluate_algo3_results_file(input_path, output_path)
