from pathlib import Path

import pandas as pd
import pytest

from llm_conceptual_modeling.algo1.evaluation import evaluate_results_file as eval_algo1
from llm_conceptual_modeling.algo1.factorial import run_factorial_analysis as factorial_algo1
from llm_conceptual_modeling.algo3.evaluation import evaluate_results_file as eval_algo3


def test_algo1_evaluation_rejects_missing_required_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "invalid.csv"
    output_path = tmp_path / "out.csv"
    pd.DataFrame({"graph": ["[]"]}).to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        eval_algo1(input_path, output_path)


def test_algo1_factorial_rejects_missing_metric_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "invalid.csv"
    output_path = tmp_path / "out.csv"
    pd.DataFrame(
        {
            "Explanation": [1],
            "Example": [1],
            "Counterexample": [1],
            "Array/List(1/-1)": [1],
            "Tag/Adjacency(1/-1)": [1],
        }
    ).to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="Missing required metric columns"):
        factorial_algo1([input_path], output_path)


def test_algo3_evaluation_rejects_missing_required_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "invalid.csv"
    output_path = tmp_path / "out.csv"
    pd.DataFrame({"Source Graph": ["[]"]}).to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        eval_algo3(input_path, output_path)
