import json

from llm_conceptual_modeling.cli import main


def test_cli_generate_algo1_stub_returns_structured_not_implemented_payload(capsys) -> None:
    exit_code = main(["generate", "algo1", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["algorithm"] == "algo1"
    assert payload["mode"] == "offline-manifest"
    assert payload["implemented"] is False
    assert payload["requires_live_llm"] is True
    assert payload["input_data"]["categories_csv"].endswith(
        "Giabbanelli & Macewan (categories).csv"
    )
    assert payload["condition_count"] == 32
    assert payload["subgraph_pairs"] == ["sg1_sg2", "sg2_sg3", "sg3_sg1"]
    assert "recommend more links" in payload["prompt_preview"]


def test_cli_generate_algo3_stub_accepts_fixture_only_mode(capsys) -> None:
    exit_code = main(["generate", "algo3", "--fixture-only", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["algorithm"] == "algo3"
    assert payload["fixture_only"] is True
    assert payload["next_step"] == "provide_fixture_dataset"
    assert payload["condition_count"] == 16
    assert payload["replications"] == 5
    assert payload["subgraph_pairs"] == [
        "subgraph_1_to_subgraph_3",
        "subgraph_2_to_subgraph_1",
        "subgraph_2_to_subgraph_3",
    ]
    assert "recommend" in payload["prompt_preview"]
