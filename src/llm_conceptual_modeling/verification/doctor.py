from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from llm_conceptual_modeling.algo1.evaluation import (
    evaluate_results_file as eval_algo1,
)
from llm_conceptual_modeling.algo1.generation import (
    build_generation_manifest as build_algo1_manifest,
)
from llm_conceptual_modeling.algo2.evaluation import (
    evaluate_results_file as eval_algo2,
)
from llm_conceptual_modeling.algo2.generation import (
    build_generation_manifest as build_algo2_manifest,
)
from llm_conceptual_modeling.algo3.generation import (
    build_generation_manifest as build_algo3_manifest,
)
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


def _build_manifest_checks() -> list[dict[str, object]]:
    algo1_manifest: dict[str, object] = build_algo1_manifest(fixture_only=False)
    algo2_manifest: dict[str, object] = build_algo2_manifest(fixture_only=False)
    algo3_manifest: dict[str, object] = build_algo3_manifest(fixture_only=False)
    mc1 = algo1_manifest.get("method_contract") or {}
    mc2 = algo2_manifest.get("method_contract") or {}
    mc3 = algo3_manifest.get("method_contract") or {}
    algo1_method_contract: dict[str, object] = mc1
    algo2_method_contract: dict[str, object] = mc2
    algo3_method_contract: dict[str, object] = mc3
    return [
        _check(
            "algo1_manifest",
            algo1_manifest["implemented"] is True
            and algo1_manifest["requires_live_llm"] is True
            and algo1_method_contract.get("uses_chain_of_verification") is True,
            {
                "implemented": algo1_manifest["implemented"],
                "requires_live_llm": algo1_manifest["requires_live_llm"],
                "prompt_preview": algo1_manifest["prompt_preview"],
            },
        ),
        _check(
            "algo2_manifest",
            algo2_manifest["implemented"] is True and algo2_manifest["requires_live_llm"] is True,
            {
                "implemented": algo2_manifest["implemented"],
                "requires_live_llm": algo2_manifest["requires_live_llm"],
                "convergence_rule": algo2_method_contract["convergence_rule"],
                "embedding_model": algo2_method_contract["embedding_model"],
                "embedding_provider": algo2_method_contract["embedding_provider"],
                "paper_embedding_model": algo2_method_contract["paper_embedding_model"],
                "paper_embedding_provider": algo2_method_contract["paper_embedding_provider"],
                "convergence_threshold_levels": algo2_method_contract[
                    "convergence_threshold_levels"
                ],
                "thesaurus_path": algo2_method_contract["thesaurus_path"],
            },
        ),
        _check(
            "algo2_convergence_threshold_levels",
            algo2_method_contract["convergence_threshold_levels"] == [0.01, 0.02],
            {
                "value": algo2_method_contract["convergence_threshold_levels"],
                "rule": algo2_method_contract["convergence_rule"],
            },
        ),
        _check(
            "algo2_embedding_model",
            algo2_method_contract["embedding_model"] == "mistral-embed-2312",
            {
                "value": algo2_method_contract["embedding_model"],
            },
        ),
        _check(
            "algo2_paper_embedding_model",
            algo2_method_contract["paper_embedding_model"] == "text-embedding-3-large"
            and algo2_method_contract["paper_embedding_provider"] == "openrouter",
            {
                "paper_embedding_model": algo2_method_contract["paper_embedding_model"],
                "paper_embedding_provider": algo2_method_contract["paper_embedding_provider"],
            },
        ),
        _check(
            "algo2_thesaurus",
            Path(str(algo2_method_contract["thesaurus_path"])).exists(),
            {
                "path": algo2_method_contract["thesaurus_path"],
                "synonym_entry_count": algo2_method_contract["synonym_entry_count"],
                "antonym_entry_count": algo2_method_contract["antonym_entry_count"],
            },
        ),
        _check(
            "algo3_manifest",
            algo3_manifest["implemented"] is True
            and algo3_manifest["requires_live_llm"] is True
            and algo3_method_contract["child_count_levels"] == [3, 5]
            and algo3_method_contract["depth_levels"] == [1, 2],
            {
                "implemented": algo3_manifest["implemented"],
                "requires_live_llm": algo3_manifest["requires_live_llm"],
                "child_count_levels": algo3_method_contract["child_count_levels"],
                "depth_levels": algo3_method_contract["depth_levels"],
            },
        ),
    ]


def _build_schema_checks(temp_root: Path) -> list[dict[str, object]]:
    algo1_output = temp_root / "algo1_eval.csv"
    algo2_output = temp_root / "algo2_eval.csv"
    algo1_raw = FIXTURES_ROOT / "algo1" / "gpt-5" / "raw" / "algorithm1_results_sg1_sg2.csv"
    algo2_raw = FIXTURES_ROOT / "algo2" / "gpt-5" / "raw" / "algorithm2_results_sg1_sg2.csv"

    eval_algo1(algo1_raw, algo1_output)
    eval_algo2(algo2_raw, algo2_output)

    algo1_columns = list(pd.read_csv(algo1_output).columns)
    algo2_columns = list(pd.read_csv(algo2_output).columns)
    return [
        _check(
            "algo1_f1_schema",
            "f1" in algo1_columns,
            {"columns": algo1_columns},
        ),
        _check(
            "algo2_f1_schema",
            "f1" in algo2_columns,
            {"columns": algo2_columns},
        ),
    ]


def _check(name: str, passed: bool, evidence: dict[str, object]) -> dict[str, object]:
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "evidence": evidence,
    }
