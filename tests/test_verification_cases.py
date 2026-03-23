from llm_conceptual_modeling.verification_cases import build_legacy_parity_cases


def test_build_legacy_parity_cases_covers_all_verified_workflows() -> None:
    cases = build_legacy_parity_cases()

    assert [case.name for case in cases] == [
        "algo1-eval",
        "algo2-eval",
        "algo3-eval",
        "algo1-factorial",
        "algo2-factorial",
        "algo3-factorial",
    ]
    assert cases[0].raw_path.as_posix().endswith(
        "tests/fixtures/legacy/algo1/gpt-5/raw/algorithm1_results_sg1_sg2.csv"
    )
    assert cases[2].expected_path.as_posix().endswith(
        "tests/fixtures/legacy/algo3/gpt-5/evaluated/method3_results_evaluated_gpt5.csv"
    )
    assert cases[3].input_paths is not None
    assert len(cases[3].input_paths) == 3
    assert cases[5].input_path is not None
