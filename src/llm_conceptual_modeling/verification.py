import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from llm_conceptual_modeling.algo1.evaluation import evaluate_results_file as eval_algo1
from llm_conceptual_modeling.algo1.factorial import run_factorial_analysis as factorial_algo1
from llm_conceptual_modeling.algo2.evaluation import evaluate_results_file as eval_algo2
from llm_conceptual_modeling.algo2.factorial import run_factorial_analysis as factorial_algo2
from llm_conceptual_modeling.algo3.evaluation import evaluate_results_file as eval_algo3
from llm_conceptual_modeling.algo3.factorial import run_factorial_analysis as factorial_algo3
from llm_conceptual_modeling.common.types import VerificationResult

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_ROOT = REPO_ROOT / "tests" / "fixtures" / "legacy"


def build_doctor_report() -> dict[str, object]:
    fixtures_present = FIXTURES_ROOT.exists()
    package_import = True
    return {
        "status": "ok" if fixtures_present and package_import else "error",
        "checks": {
            "fixtures_present": fixtures_present,
            "package_import": package_import,
        },
    }


def run_legacy_parity_verification() -> dict[str, object]:
    results: list[VerificationResult] = []
    with TemporaryDirectory() as tmpdir:
        temp_root = Path(tmpdir)

        _verify_eval(
            "algo1-eval",
            FIXTURES_ROOT / "algo1" / "gpt-5" / "raw" / "algorithm1_results_sg1_sg2.csv",
            FIXTURES_ROOT / "algo1" / "gpt-5" / "evaluated" / "metrics_sg1_sg2.csv",
            temp_root / "algo1_metrics.csv",
            eval_algo1,
            "accuracy",
            results,
        )
        _verify_eval(
            "algo2-eval",
            FIXTURES_ROOT / "algo2" / "gpt-5" / "raw" / "algorithm2_results_sg1_sg2.csv",
            FIXTURES_ROOT / "algo2" / "gpt-5" / "evaluated" / "metrics_sg1_sg2.csv",
            temp_root / "algo2_metrics.csv",
            eval_algo2,
            "accuracy",
            results,
        )
        _verify_eval(
            "algo3-eval",
            FIXTURES_ROOT / "algo3" / "gpt-5" / "raw" / "method3_results_gpt5.csv",
            FIXTURES_ROOT / "algo3" / "gpt-5" / "evaluated" / "method3_results_evaluated_gpt5.csv",
            temp_root / "algo3_metrics.csv",
            eval_algo3,
            "Recall",
            results,
        )
        _verify_factorial(
            "algo1-factorial",
            [
                FIXTURES_ROOT / "algo1" / "gpt-5" / "evaluated" / "metrics_sg1_sg2.csv",
                FIXTURES_ROOT / "algo1" / "gpt-5" / "evaluated" / "metrics_sg2_sg3.csv",
                FIXTURES_ROOT / "algo1" / "gpt-5" / "evaluated" / "metrics_sg3_sg1.csv",
            ],
            FIXTURES_ROOT
            / "algo1"
            / "gpt-5"
            / "factorial"
            / "factorial_analysis_algo1_gpt_5_without_error.csv",
            temp_root / "algo1_factorial.csv",
            factorial_algo1,
            results,
        )
        _verify_factorial(
            "algo2-factorial",
            [
                FIXTURES_ROOT / "algo2" / "gpt-5" / "evaluated" / "metrics_sg1_sg2.csv",
                FIXTURES_ROOT / "algo2" / "gpt-5" / "evaluated" / "metrics_sg2_sg3.csv",
                FIXTURES_ROOT / "algo2" / "gpt-5" / "evaluated" / "metrics_sg3_sg1.csv",
            ],
            FIXTURES_ROOT
            / "algo2"
            / "gpt-5"
            / "factorial"
            / "factorial_analysis_gpt_5_algo2_without_error.csv.csv",
            temp_root / "algo2_factorial.csv",
            factorial_algo2,
            results,
        )
        _verify_factorial(
            "algo3-factorial",
            FIXTURES_ROOT / "algo3" / "gpt-5" / "evaluated" / "method3_results_evaluated_gpt5.csv",
            FIXTURES_ROOT
            / "algo3"
            / "gpt-5"
            / "factorial"
            / "factorial_analysis_results_gpt5_without_error.csv",
            temp_root / "algo3_factorial.csv",
            factorial_algo3,
            results,
        )

    status = "ok" if all(result.status == "passed" for result in results) else "error"
    return {"status": status, "results": [result.to_dict() for result in results]}


def run_full_verification() -> dict[str, object]:
    doctor = build_doctor_report()
    legacy_parity = run_legacy_parity_verification()
    status = "ok" if doctor["status"] == "ok" and legacy_parity["status"] == "ok" else "error"
    return {
        "status": status,
        "doctor": doctor,
        "legacy_parity": legacy_parity,
    }


def emit_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload))


def _verify_eval(
    name: str,
    raw_path: Path,
    expected_path: Path,
    output_path: Path,
    evaluator,
    metric_column: str,
    results: list[VerificationResult],
) -> None:
    evaluator(raw_path, output_path)
    actual = pd.read_csv(output_path)
    expected = pd.read_csv(expected_path)
    status = "passed" if actual[metric_column].equals(expected[metric_column]) else "failed"
    results.append(VerificationResult(name=name, status=status))


def _verify_factorial(
    name: str,
    input_paths,
    expected_path: Path,
    output_path: Path,
    evaluator,
    results: list[VerificationResult],
) -> None:
    evaluator(input_paths, output_path)
    actual = pd.read_csv(output_path)
    expected = pd.read_csv(expected_path)
    status = "passed" if actual.equals(expected) else "failed"
    results.append(VerificationResult(name=name, status=status))
