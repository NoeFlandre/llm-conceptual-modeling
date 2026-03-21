import pandas as pd

from llm_conceptual_modeling.algo1.evaluation import evaluate_results_file as eval_algo1
from llm_conceptual_modeling.algo1.factorial import run_factorial_analysis as factorial_algo1
from llm_conceptual_modeling.algo3.factorial import run_factorial_analysis as factorial_algo3


def test_algo1_evaluation_output_schema(tmp_path) -> None:
    output_path = tmp_path / "metrics.csv"
    eval_algo1("tests/fixtures/legacy/algo1/gpt-5/raw/algorithm1_results_sg1_sg2.csv", output_path)

    actual = pd.read_csv(output_path)
    assert list(actual.columns) == [
        "Repetition",
        "Result",
        "Explanation",
        "Example",
        "Counterexample",
        "Array/List(1/-1)",
        "Tag/Adjacency(1/-1)",
        "subgraph1",
        "subgraph2",
        "graph",
        "accuracy",
        "recall",
        "precision",
        "f1",
    ]


def test_algo1_factorial_output_schema(tmp_path) -> None:
    output_path = tmp_path / "factorial.csv"
    factorial_algo1(
        [
            "tests/fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg1_sg2.csv",
            "tests/fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg2_sg3.csv",
            "tests/fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg3_sg1.csv",
        ],
        output_path,
    )

    actual = pd.read_csv(output_path)
    assert list(actual.columns) == ["accuracy", "recall", "precision", "Feature"]


def test_algo3_factorial_output_schema(tmp_path) -> None:
    output_path = tmp_path / "factorial.csv"
    factorial_algo3(
        "tests/fixtures/legacy/algo3/gpt-5/evaluated/method3_results_evaluated_gpt5.csv",
        output_path,
    )

    actual = pd.read_csv(output_path)
    assert list(actual.columns) == ["Feature", "Result"]
