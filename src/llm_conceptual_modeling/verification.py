import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import pandas as pd

from llm_conceptual_modeling.algo1.evaluation import evaluate_results_file as eval_algo1
from llm_conceptual_modeling.algo1.experiment import build_algo1_experiment_specs
from llm_conceptual_modeling.algo1.factorial import run_factorial_analysis as factorial_algo1
from llm_conceptual_modeling.algo1.generation import (
    build_generation_manifest as build_algo1_manifest,
)
from llm_conceptual_modeling.algo2.evaluation import evaluate_results_file as eval_algo2
from llm_conceptual_modeling.algo2.experiment import build_algo2_experiment_specs
from llm_conceptual_modeling.algo2.factorial import run_factorial_analysis as factorial_algo2
from llm_conceptual_modeling.algo2.generation import (
    build_generation_manifest as build_algo2_manifest,
)
from llm_conceptual_modeling.algo3.evaluation import evaluate_results_file as eval_algo3
from llm_conceptual_modeling.algo3.experiment import build_algo3_experiment_specs
from llm_conceptual_modeling.algo3.factorial import run_factorial_analysis as factorial_algo3
from llm_conceptual_modeling.algo3.generation import (
    build_generation_manifest as build_algo3_manifest,
)
from llm_conceptual_modeling.common.types import VerificationResult
from llm_conceptual_modeling.post_revision_debug.run_context import ProbeRunContext

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
            / "factorial_analysis_gpt_5_algo2_without_error.csv",
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


def build_paper_alignment_report() -> dict[str, object]:
    checks: list[dict[str, object]] = []
    with TemporaryDirectory() as tmpdir:
        temp_root = Path(tmpdir)
        checks.extend(_build_manifest_checks())
        checks.extend(_build_resume_checks(temp_root))
        checks.extend(_build_schema_checks(temp_root))
        checks.append(_build_probe_context_check(temp_root))

    status = "ok" if all(check["status"] == "passed" for check in checks) else "error"
    return {
        "status": status,
        "checks": checks,
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


def _build_manifest_checks() -> list[dict[str, object]]:
    algo1_manifest: dict[str, object] = build_algo1_manifest(fixture_only=False)
    algo2_manifest: dict[str, object] = build_algo2_manifest(fixture_only=False)
    algo3_manifest: dict[str, object] = build_algo3_manifest(fixture_only=False)
    mc1 = cast(dict[str, object], algo1_manifest.get("method_contract") or {})
    mc2 = cast(dict[str, object], algo2_manifest.get("method_contract") or {})
    mc3 = cast(dict[str, object], algo3_manifest.get("method_contract") or {})
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
            algo2_manifest["implemented"] is True
            and algo2_manifest["requires_live_llm"] is True,
            {
                "implemented": algo2_manifest["implemented"],
                "requires_live_llm": algo2_manifest["requires_live_llm"],
                "convergence_rule": algo2_method_contract["convergence_rule"],
                "embedding_model": algo2_method_contract["embedding_model"],
                "convergence_threshold": algo2_method_contract["convergence_threshold"],
                "thesaurus_path": algo2_method_contract["thesaurus_path"],
            },
        ),
        _check(
            "algo2_convergence_threshold",
            algo2_method_contract["convergence_threshold"] == 0.01,
            {
                "value": algo2_method_contract["convergence_threshold"],
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
            "algo2_thesaurus",
            Path(cast(str, algo2_method_contract["thesaurus_path"])).exists(),
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


def _build_resume_checks(temp_root: Path) -> list[dict[str, object]]:
    algo1_spec = build_algo1_experiment_specs(
        pair_name="sg1_sg2",
        model="verification",
        output_root=temp_root / "algo1",
        replications=1,
        resume=True,
    )[0]
    algo2_spec = build_algo2_experiment_specs(
        pair_name="sg1_sg2",
        output_root=temp_root / "algo2",
        replications=1,
        resume=True,
    )[0]
    algo3_spec = build_algo3_experiment_specs(
        pair_name="subgraph_1_to_subgraph_3",
        output_root=temp_root / "algo3",
        replications=1,
        resume=True,
    )[0]
    return [
        _check("algo1_resume_support", algo1_spec.resume is True, {"resume": algo1_spec.resume}),
        _check("algo2_resume_support", algo2_spec.resume is True, {"resume": algo2_spec.resume}),
        _check("algo3_resume_support", algo3_spec.resume is True, {"resume": algo3_spec.resume}),
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


def _build_probe_context_check(temp_root: Path) -> dict[str, object]:
    context = ProbeRunContext(
        output_dir=temp_root / "probe_run",
        run_name="paper_alignment_audit",
        algorithm="algo2",
    )
    context.record_manifest({"run_name": "paper_alignment_audit"})
    context.record_prompt("prompt.txt", "prompt", stage="prompt_written")
    context.record_checkpoint(
        "checkpoint.json",
        {"status": "done"},
        stage="checkpoint_written",
    )
    state = context.load_state()
    return _check(
        "probe_context_checkpointing",
        context.manifest_path.exists()
        and context.summary_path.exists() is False
        and context.state_path.exists()
        and context.log_path.exists()
        and context.events_path.exists() is False
        and state["completed_stages"] == [
            "manifest_written",
            "prompt_written",
            "checkpoint_written",
        ],
        {
            "manifest_path": str(context.manifest_path),
            "state_path": str(context.state_path),
            "log_path": str(context.log_path),
            "completed_stages": state["completed_stages"],
        },
    )


def _check(name: str, passed: bool, evidence: dict[str, object]) -> dict[str, object]:
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "evidence": evidence,
    }
