import json

from llm_conceptual_modeling.cli import main


def test_cli_doctor_reports_ok_status(capsys) -> None:
    exit_code = main(["doctor", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["checks"]["fixtures_present"] is True
    assert payload["checks"]["package_import"] is True


def test_cli_verify_legacy_parity_reports_all_workflows_green(capsys) -> None:
    exit_code = main(["verify", "legacy-parity", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["status"] == "ok"
    assert all(result["status"] == "passed" for result in payload["results"])
    assert len(payload["results"]) == 6
