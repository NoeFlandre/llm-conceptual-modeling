from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from llm_conceptual_modeling.algo1.evaluation import evaluate_results_file as eval_algo1
from llm_conceptual_modeling.algo1.factorial import run_factorial_analysis as factorial_algo1
from llm_conceptual_modeling.algo2.evaluation import evaluate_results_file as eval_algo2
from llm_conceptual_modeling.algo2.factorial import run_factorial_analysis as factorial_algo2
from llm_conceptual_modeling.algo3.evaluation import evaluate_results_file as eval_algo3
from llm_conceptual_modeling.algo3.factorial import run_factorial_analysis as factorial_algo3

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_ROOT = REPO_ROOT / "tests" / "reference_fixtures" / "legacy"


@dataclass(frozen=True)
class VerificationCase:
    name: str
    evaluator: Callable[..., None]
    expected_path: Path
    output_path: Path
    raw_path: Path | None = None
    input_path: Path | None = None
    input_paths: tuple[Path, ...] | None = None
    metric_column: str | None = None


def build_legacy_parity_cases(temp_root: Path | None = None) -> list[VerificationCase]:
    root = temp_root or Path("/tmp")
    return [
        VerificationCase(
            name="algo1-eval",
            evaluator=eval_algo1,
            raw_path=FIXTURES_ROOT / "algo1" / "gpt-5" / "raw" / "algorithm1_results_sg1_sg2.csv",
            expected_path=FIXTURES_ROOT / "algo1" / "gpt-5" / "evaluated" / "metrics_sg1_sg2.csv",
            output_path=root / "algo1_metrics.csv",
            metric_column="accuracy",
        ),
        VerificationCase(
            name="algo2-eval",
            evaluator=eval_algo2,
            raw_path=FIXTURES_ROOT / "algo2" / "gpt-5" / "raw" / "algorithm2_results_sg1_sg2.csv",
            expected_path=FIXTURES_ROOT / "algo2" / "gpt-5" / "evaluated" / "metrics_sg1_sg2.csv",
            output_path=root / "algo2_metrics.csv",
            metric_column="accuracy",
        ),
        VerificationCase(
            name="algo3-eval",
            evaluator=eval_algo3,
            raw_path=FIXTURES_ROOT / "algo3" / "gpt-5" / "raw" / "method3_results_gpt5.csv",
            expected_path=FIXTURES_ROOT
            / "algo3"
            / "gpt-5"
            / "evaluated"
            / "method3_results_evaluated_gpt5.csv",
            output_path=root / "algo3_metrics.csv",
            metric_column="Recall",
        ),
        VerificationCase(
            name="algo1-factorial",
            evaluator=factorial_algo1,
            input_paths=(
                FIXTURES_ROOT / "algo1" / "gpt-5" / "evaluated" / "metrics_sg1_sg2.csv",
                FIXTURES_ROOT / "algo1" / "gpt-5" / "evaluated" / "metrics_sg2_sg3.csv",
                FIXTURES_ROOT / "algo1" / "gpt-5" / "evaluated" / "metrics_sg3_sg1.csv",
            ),
            expected_path=FIXTURES_ROOT
            / "algo1"
            / "gpt-5"
            / "factorial"
            / "factorial_analysis_algo1_gpt_5_without_error.csv",
            output_path=root / "algo1_factorial.csv",
        ),
        VerificationCase(
            name="algo2-factorial",
            evaluator=factorial_algo2,
            input_paths=(
                FIXTURES_ROOT / "algo2" / "gpt-5" / "evaluated" / "metrics_sg1_sg2.csv",
                FIXTURES_ROOT / "algo2" / "gpt-5" / "evaluated" / "metrics_sg2_sg3.csv",
                FIXTURES_ROOT / "algo2" / "gpt-5" / "evaluated" / "metrics_sg3_sg1.csv",
            ),
            expected_path=FIXTURES_ROOT
            / "algo2"
            / "gpt-5"
            / "factorial"
            / "factorial_analysis_gpt_5_algo2_without_error.csv",
            output_path=root / "algo2_factorial.csv",
        ),
        VerificationCase(
            name="algo3-factorial",
            evaluator=factorial_algo3,
            input_path=FIXTURES_ROOT
            / "algo3"
            / "gpt-5"
            / "evaluated"
            / "method3_results_evaluated_gpt5.csv",
            expected_path=FIXTURES_ROOT
            / "algo3"
            / "gpt-5"
            / "factorial"
            / "factorial_analysis_results_gpt5_without_error.csv",
            output_path=root / "algo3_factorial.csv",
        ),
    ]


def run_verification_case(case: VerificationCase) -> bool:
    if case.input_paths is not None:
        case.evaluator(case.input_paths, case.output_path)
    elif case.input_path is not None:
        case.evaluator(case.input_path, case.output_path)
    elif case.raw_path is not None:
        case.evaluator(case.raw_path, case.output_path)
    else:
        raise ValueError(f"Verification case {case.name} is missing input paths")

    actual = pd.read_csv(case.output_path)
    expected = pd.read_csv(case.expected_path)

    if case.metric_column is not None:
        return actual[case.metric_column].equals(expected[case.metric_column])
    return actual.equals(expected)
