import json
from pathlib import Path

from llm_conceptual_modeling.cli import main

SNAPSHOT_ROOT = Path("tests/fixtures/snapshots")


def _read_snapshot(name: str) -> dict[str, object]:
    return json.loads((SNAPSHOT_ROOT / name).read_text())


def test_doctor_json_snapshot(capsys) -> None:
    exit_code = main(["doctor", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload == _read_snapshot("doctor.json")


def test_generate_algo1_json_snapshot(capsys) -> None:
    exit_code = main(["generate", "algo1", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    payload["input_data"]["categories_csv"] = "data/inputs/Giabbanelli & Macewan (categories).csv"
    payload["input_data"]["edges_csv"] = "data/inputs/Giabbanelli & Macewan (edges).csv"
    assert payload == _read_snapshot("generate_algo1.json")


def test_verify_all_json_snapshot(capsys) -> None:
    exit_code = main(["verify", "all", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload == _read_snapshot("verify_all.json")
