import json

from llm_conceptual_modeling.cli import main


def test_cli_audit_paper_alignment_reports_core_contracts(capsys) -> None:
    exit_code = main(["audit", "paper-alignment", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["status"] == "ok"

    checks = {check["name"]: check for check in payload["checks"]}

    assert checks["algo1_manifest"]["status"] == "passed"
    assert checks["algo2_manifest"]["status"] == "passed"
    assert checks["algo3_manifest"]["status"] == "passed"
    assert checks["algo1_resume_support"]["status"] == "passed"
    assert checks["algo2_resume_support"]["status"] == "passed"
    assert checks["algo3_resume_support"]["status"] == "passed"
    assert checks["algo2_manifest"]["status"] == "passed"
    assert checks["algo2_manifest"]["evidence"]["convergence_threshold"] == 0.01
    assert checks["algo2_manifest"]["evidence"]["embedding_model"] == "mistral-embed-2312"
    assert checks["algo2_convergence_threshold"]["status"] == "passed"
    assert checks["algo2_convergence_threshold"]["evidence"]["value"] == 0.01
    assert checks["algo2_embedding_model"]["status"] == "passed"
    assert checks["algo2_embedding_model"]["evidence"]["value"] == "mistral-embed-2312"
    assert checks["algo2_thesaurus"]["status"] == "passed"
    assert checks["algo1_f1_schema"]["status"] == "passed"
    assert checks["algo2_f1_schema"]["status"] == "passed"
    assert checks["probe_context_checkpointing"]["status"] == "passed"
