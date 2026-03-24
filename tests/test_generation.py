import json

from llm_conceptual_modeling.cli import main


def test_cli_generate_algo1_stub_returns_structured_not_implemented_payload(capsys) -> None:
    exit_code = main(["generate", "algo1", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["algorithm"] == "algo1"
    assert payload["mode"] == "offline-manifest"
    assert payload["implemented"] is True
    assert payload["requires_live_llm"] is True
    assert payload["next_step"] == "provide_model_pair_and_output_root"
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
    assert payload["implemented"] is True
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
    assert method_contract["convergence_threshold_levels"] == [0.01, 0.02]
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
