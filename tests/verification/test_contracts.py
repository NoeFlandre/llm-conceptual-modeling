import json

import pandas as pd

from llm_conceptual_modeling.algo1.evaluation import evaluate_results_file as eval_algo1
from llm_conceptual_modeling.algo2.evaluation import evaluate_results_file as eval_algo2
from llm_conceptual_modeling.algo3.evaluation import evaluate_results_file as eval_algo3
from llm_conceptual_modeling.cli import main


def test_algo1_evaluation_matches_full_legacy_csv(tmp_path) -> None:
    raw_path = "tests/reference_fixtures/legacy/algo1/gpt-5/raw/algorithm1_results_sg1_sg2.csv"
    expected_path = "tests/reference_fixtures/legacy/algo1/gpt-5/evaluated/metrics_sg1_sg2.csv"
    output_path = tmp_path / "algo1_metrics.csv"

    eval_algo1(raw_path, output_path)

    pd.testing.assert_frame_equal(pd.read_csv(output_path), pd.read_csv(expected_path))


def test_algo2_evaluation_matches_full_legacy_csv(tmp_path) -> None:
    raw_path = "tests/reference_fixtures/legacy/algo2/gpt-5/raw/algorithm2_results_sg1_sg2.csv"
    expected_path = "tests/reference_fixtures/legacy/algo2/gpt-5/evaluated/metrics_sg1_sg2.csv"
    output_path = tmp_path / "algo2_metrics.csv"

    eval_algo2(raw_path, output_path)

    pd.testing.assert_frame_equal(pd.read_csv(output_path), pd.read_csv(expected_path))


def test_algo3_evaluation_matches_full_legacy_csv(tmp_path) -> None:
    raw_path = "tests/reference_fixtures/legacy/algo3/gpt-5/raw/method3_results_gpt5.csv"
    expected_path = (
        "tests/reference_fixtures/legacy/algo3/gpt-5/evaluated/method3_results_evaluated_gpt5.csv"
    )
    output_path = tmp_path / "algo3_metrics.csv"

    eval_algo3(raw_path, output_path)

    pd.testing.assert_frame_equal(pd.read_csv(output_path), pd.read_csv(expected_path))


def test_cli_verify_all_reports_ok_status(capsys) -> None:
    exit_code = main(["verify", "all", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["doctor"]["status"] == "ok"
    assert payload["legacy_parity"]["status"] == "ok"
