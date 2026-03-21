import json

from llm_conceptual_modeling.cli import main


def test_cli_generate_algo1_stub_returns_structured_not_implemented_payload(capsys) -> None:
    exit_code = main(["generate", "algo1", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["algorithm"] == "algo1"
    assert payload["mode"] == "stub"
    assert payload["implemented"] is False
    assert payload["requires_live_llm"] is True


def test_cli_generate_algo3_stub_accepts_fixture_only_mode(capsys) -> None:
    exit_code = main(["generate", "algo3", "--fixture-only", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["algorithm"] == "algo3"
    assert payload["fixture_only"] is True
    assert payload["next_step"] == "provide_fixture_dataset"
