from llm_conceptual_modeling.verification.doctor import (
    build_doctor_report,
    emit_json,
    run_full_verification,
    run_legacy_parity_verification,
)


def test_verification_package_imports() -> None:
    assert callable(build_doctor_report)
    assert callable(emit_json)
    assert callable(run_full_verification)
    assert callable(run_legacy_parity_verification)
