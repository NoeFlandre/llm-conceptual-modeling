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


def test_cli_generate_algo3_exposes_paper_method_contract(capsys) -> None:
    exit_code = main(["generate", "algo3", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    method_contract = payload["method_contract"]

    assert exit_code == 0
    assert payload["algorithm"] == "algo3"
    assert method_contract["method_name"] == "Tree-based approach for indirect matches"
    assert method_contract["phases"] == [
        "source_tree_expansion",
        "target_match_check",
        "recursive_expansion",
    ]
    assert method_contract["child_count_levels"] == [3, 5]
    assert method_contract["depth_levels"] == [1, 2]
    assert method_contract["stops_descending_from_matched_words"] is True
    assert method_contract["expansion_scope"] == "source_labels_only"


def test_cli_generate_algo2_exposes_paper_method_contract(capsys) -> None:
    exit_code = main(["generate", "algo2", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    method_contract = payload["method_contract"]

    assert exit_code == 0
    assert payload["algorithm"] == "algo2"
    expected_method_name = "Grow candidate links through creative thinking and word embeddings"

    assert method_contract["method_name"] == expected_method_name
    assert method_contract["phases"] == [
        "label_expansion",
        "edge_suggestion",
        "thesaurus_normalization",
        "chain_of_verification",
    ]
    assert method_contract["embedding_model"] == "mistral-embed-2312"
    assert method_contract["convergence_rule"] == "absolute_cosine_similarity_delta <= threshold"
    assert method_contract["paper_confirmed_threshold"] == 0.01
    assert method_contract["paper_unresolved_variant"] == "half_as_early"
    assert method_contract["uses_domain_thesaurus"] is True
    assert method_contract["uses_chain_of_verification"] is True


def test_cli_generate_algo1_exposes_paper_method_contract(capsys) -> None:
    exit_code = main(["generate", "algo1", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    method_contract = payload["method_contract"]

    assert exit_code == 0
    assert payload["algorithm"] == "algo1"
    assert method_contract["method_name"] == "Direct combination"
    assert method_contract["phases"] == [
        "edge_generation",
        "chain_of_verification",
    ]
    assert method_contract["uses_chain_of_verification"] is True
    assert method_contract["verification_output"] == "Y/N list aligned to candidate pairs"
    assert method_contract["allows_new_nodes"] is True


def test_cli_generate_algo1_can_execute_experiment_specs(monkeypatch, capsys, tmp_path) -> None:
    captured_call: dict[str, object] = {}

    class FakeChatClient:
        def __init__(self, *, api_key: str, model: str) -> None:
            captured_call["api_key"] = api_key
            captured_call["model"] = model

    def fake_build_specs(*, pair_name, output_root, replications):
        captured_call["pair_name"] = pair_name
        captured_call["output_root"] = str(output_root)
        captured_call["replications"] = replications
        return ["spec-a", "spec-b"]

    def fake_run_experiment(*, specs, chat_client):
        captured_call["specs"] = specs
        captured_call["chat_client_type"] = type(chat_client).__name__
        return [
            {"run_name": "spec-a", "verified_edges": [["alpha", "beta"]]},
            {"run_name": "spec-b", "verified_edges": [["gamma", "delta"]]},
        ]

    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.generate.Algo1ChatClient",
        FakeChatClient,
    )
    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.generate.build_algo1_experiment_specs",
        fake_build_specs,
    )
    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.generate.run_algo1_experiment",
        fake_run_experiment,
    )

    exit_code = main(
        [
            "generate",
            "algo1",
            "--model",
            "mistral-small-2603",
            "--pair",
            "sg1_sg2",
            "--output-root",
            str(tmp_path / "runs"),
            "--replications",
            "2",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["algorithm"] == "algo1"
    assert payload["mode"] == "executed-experiment"
    assert payload["result_count"] == 2
    assert captured_call["api_key"] == "test-key"
    assert captured_call["model"] == "mistral-small-2603"
    assert captured_call["pair_name"] == "sg1_sg2"
    assert captured_call["replications"] == 2
    assert captured_call["specs"] == ["spec-a", "spec-b"]
    assert captured_call["chat_client_type"] == "FakeChatClient"
