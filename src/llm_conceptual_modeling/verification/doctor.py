from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from llm_conceptual_modeling.common.types import VerificationResult
from llm_conceptual_modeling.hf_batch.monitoring import collect_batch_status
from llm_conceptual_modeling.verification.cases import (
    FIXTURES_ROOT,
    build_legacy_parity_cases,
    run_verification_case,
)


def build_doctor_report(
    *,
    results_root: str | None = None,
    smoke_root: str | None = None,
) -> dict[str, object]:
    fixtures_present = FIXTURES_ROOT.exists()
    package_import = True
    checks: dict[str, object] = {
        "fixtures_present": fixtures_present,
        "package_import": package_import,
    }
    if results_root is not None:
        results_root_path = Path(results_root)
        results_root_path.mkdir(parents=True, exist_ok=True)
        checks["results_root_writable"] = results_root_path.exists()
        checks["batch_status"] = collect_batch_status(results_root_path)
    if smoke_root is not None:
        smoke_root_path = Path(smoke_root)
        smoke_verdict_path = smoke_root_path / "smoke_verdict.json"
        checks["smoke_verdict_present"] = smoke_verdict_path.exists()
        if smoke_verdict_path.exists():
            checks["smoke_verdict"] = json.loads(smoke_verdict_path.read_text(encoding="utf-8"))
    bootstrap_snapshot_path = Path.cwd() / ".bootstrap-runtime.json"
    if bootstrap_snapshot_path.exists():
        checks["bootstrap_snapshot"] = json.loads(
            bootstrap_snapshot_path.read_text(encoding="utf-8")
        )
    status = "ok" if fixtures_present and package_import else "error"
    return {
        "status": status,
        "checks": checks,
    }


def run_legacy_parity_verification() -> dict[str, object]:
    results: list[VerificationResult] = []
    with TemporaryDirectory() as tmpdir:
        temp_root = Path(tmpdir)
        cases = build_legacy_parity_cases(temp_root)
        for case in cases:
            passed = run_verification_case(case)
            results.append(
                VerificationResult(name=case.name, status="passed" if passed else "failed")
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


def _check(name: str, passed: bool, evidence: dict[str, object]) -> dict[str, object]:
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "evidence": evidence,
    }
