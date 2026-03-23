import json

from llm_conceptual_modeling.cli import main


def test_cli_probe_algo1_emits_summary_json(monkeypatch, capsys, tmp_path) -> None:
    captured_call: dict[str, object] = {}

    class FakeChatClient:
        def __init__(self, *, api_key: str, model: str) -> None:
            captured_call["api_key"] = api_key
            captured_call["model"] = model

    def fake_run_algo1_probe(*, spec, chat_client) -> dict[str, object]:
        captured_call["run_name"] = spec.run_name
        captured_call["subgraph1"] = spec.subgraph1
        captured_call["subgraph2"] = spec.subgraph2
        captured_call["output_dir"] = str(spec.output_dir)
        captured_call["chat_client_type"] = type(chat_client).__name__
        return {"run_name": spec.run_name, "verified_edges": [["alpha", "bridge"]]}

    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.probe.Algo1ChatClient",
        FakeChatClient,
    )
    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.probe.run_algo1_probe",
        fake_run_algo1_probe,
    )

    exit_code = main(
        [
            "probe",
            "algo1",
            "--run-name",
            "algo1_row0",
            "--model",
            "mistral-small-2603",
            "--subgraph1-edge",
            '["alpha", "beta"]',
            "--subgraph2-edge",
            '["gamma", "delta"]',
            "--output-dir",
            str(tmp_path / "algo1_probe"),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload == {"run_name": "algo1_row0", "verified_edges": [["alpha", "bridge"]]}
    assert captured_call["api_key"] == "test-key"
    assert captured_call["model"] == "mistral-small-2603"
    assert captured_call["run_name"] == "algo1_row0"
    assert captured_call["subgraph1"] == [("alpha", "beta")]
    assert captured_call["subgraph2"] == [("gamma", "delta")]
    assert captured_call["chat_client_type"] == "FakeChatClient"


def test_cli_probe_algo2_emits_summary_json(monkeypatch, capsys, tmp_path) -> None:
    captured_call: dict[str, object] = {}

    class FakeChatClient:
        def __init__(self, *, api_key: str, model: str) -> None:
            captured_call["chat_api_key"] = api_key
            captured_call["chat_model"] = model

    class FakeEmbeddingClient:
        def __init__(self, *, api_key: str, model: str) -> None:
            captured_call["embedding_api_key"] = api_key
            captured_call["embedding_model"] = model

    def fake_run_algo2_probe(*, spec, chat_client, embedding_client) -> dict[str, object]:
        captured_call["run_name"] = spec.run_name
        captured_call["seed_labels"] = spec.seed_labels
        captured_call["threshold"] = spec.convergence_threshold
        captured_call["embedding_provider"] = spec.embedding_provider
        captured_call["chat_client_type"] = type(chat_client).__name__
        captured_call["embedding_client_type"] = type(embedding_client).__name__
        return {"run_name": spec.run_name, "expanded_labels": ["bridge_a"]}

    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.probe.Algo2ChatClient",
        FakeChatClient,
    )
    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.probe.Algo2EmbeddingClient",
        FakeEmbeddingClient,
    )
    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.probe.run_algo2_probe",
        fake_run_algo2_probe,
    )

    exit_code = main(
        [
            "probe",
            "algo2",
            "--run-name",
            "algo2_row0",
            "--model",
            "mistral-small-2603",
            "--embedding-model",
            "mistral-embed-2312",
            "--embedding-provider",
            "mistral",
            "--seed-label",
            "alpha",
            "--seed-label",
            "beta",
            "--convergence-threshold",
            "0.01",
            "--output-dir",
            str(tmp_path / "algo2_probe"),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload == {"run_name": "algo2_row0", "expanded_labels": ["bridge_a"]}
    assert captured_call["chat_api_key"] == "test-key"
    assert captured_call["chat_model"] == "mistral-small-2603"
    assert captured_call["embedding_api_key"] == "test-key"
    assert captured_call["embedding_provider"] == "mistral"
    assert captured_call["embedding_model"] == "mistral-embed-2312"
    assert captured_call["seed_labels"] == ["alpha", "beta"]
    assert captured_call["threshold"] == 0.01
    assert captured_call["chat_client_type"] == "FakeChatClient"
    assert captured_call["embedding_client_type"] == "FakeEmbeddingClient"


def test_cli_probe_algo2_can_select_openrouter_embeddings(monkeypatch, capsys, tmp_path) -> None:
    captured_call: dict[str, object] = {}

    class FakeChatClient:
        def __init__(self, *, api_key: str, model: str) -> None:
            captured_call["chat_api_key"] = api_key
            captured_call["chat_model"] = model

    class FakeOpenRouterEmbeddingClient:
        def __init__(self, *, api_key: str, model: str) -> None:
            captured_call["embedding_api_key"] = api_key
            captured_call["embedding_model"] = model

    def fake_run_algo2_probe(*, spec, chat_client, embedding_client) -> dict[str, object]:
        captured_call["run_name"] = spec.run_name
        captured_call["embedding_provider"] = spec.embedding_provider
        captured_call["chat_client_type"] = type(chat_client).__name__
        captured_call["embedding_client_type"] = type(embedding_client).__name__
        return {"run_name": spec.run_name, "expanded_labels": ["bridge_a"]}

    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.probe.Algo2ChatClient",
        FakeChatClient,
    )
    monkeypatch.setattr(
        "llm_conceptual_modeling.commands._provider_utils.OpenRouterEmbeddingClient",
        FakeOpenRouterEmbeddingClient,
    )
    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.probe.run_algo2_probe",
        fake_run_algo2_probe,
    )

    exit_code = main(
        [
            "probe",
            "algo2",
            "--run-name",
            "algo2_row0",
            "--model",
            "mistral-small-2603",
            "--embedding-provider",
            "openrouter",
            "--embedding-model",
            "text-embedding-3-large",
            "--seed-label",
            "alpha",
            "--convergence-threshold",
            "0.01",
            "--output-dir",
            str(tmp_path / "algo2_probe"),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload == {"run_name": "algo2_row0", "expanded_labels": ["bridge_a"]}
    assert captured_call["chat_api_key"] == "mistral-key"
    assert captured_call["embedding_api_key"] == "openrouter-key"
    assert captured_call["embedding_model"] == "text-embedding-3-large"
    assert captured_call["embedding_provider"] == "openrouter"
    assert captured_call["chat_client_type"] == "FakeChatClient"
    assert captured_call["embedding_client_type"] == "FakeOpenRouterEmbeddingClient"


def test_cli_probe_algo2_resolves_paper_model_aliases(monkeypatch, capsys, tmp_path) -> None:
    captured_call: dict[str, object] = {}

    class FakeChatClient:
        def __init__(self, *, api_key: str, model: str) -> None:
            captured_call["chat_api_key"] = api_key
            captured_call["chat_model"] = model

    class FakeEmbeddingClient:
        def __init__(self, *, api_key: str, model: str) -> None:
            captured_call["embedding_api_key"] = api_key
            captured_call["embedding_model"] = model

    def fake_run_algo2_probe(*, spec, chat_client, embedding_client) -> dict[str, object]:
        captured_call["model"] = spec.model
        captured_call["embedding_model"] = spec.embedding_model
        captured_call["chat_client_type"] = type(chat_client).__name__
        captured_call["embedding_client_type"] = type(embedding_client).__name__
        return {"run_name": spec.run_name, "expanded_labels": ["bridge_a"]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setattr(
        "llm_conceptual_modeling.commands._provider_utils.OpenRouterChatClient",
        FakeChatClient,
    )
    monkeypatch.setattr(
        "llm_conceptual_modeling.commands._provider_utils.OpenRouterEmbeddingClient",
        FakeEmbeddingClient,
    )
    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.probe.run_algo2_probe",
        fake_run_algo2_probe,
    )

    exit_code = main(
        [
            "probe",
            "algo2",
            "--provider",
            "openrouter",
            "--run-name",
            "algo2_row0",
            "--model",
            "paper:gpt-5",
            "--embedding-provider",
            "openrouter",
            "--embedding-model",
            "paper:text-embedding-3-large",
            "--seed-label",
            "alpha",
            "--convergence-threshold",
            "0.01",
            "--output-dir",
            str(tmp_path / "algo2_probe"),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload == {"run_name": "algo2_row0", "expanded_labels": ["bridge_a"]}
    assert captured_call["chat_api_key"] == "openrouter-key"
    assert captured_call["embedding_api_key"] == "openrouter-key"
    assert captured_call["chat_model"] == "openai/gpt-5"
    assert captured_call["embedding_model"] == "text-embedding-3-large"
    assert captured_call["model"] == "openai/gpt-5"
    assert captured_call["embedding_model"] == "text-embedding-3-large"
    assert captured_call["chat_client_type"] == "FakeChatClient"
    assert captured_call["embedding_client_type"] == "FakeEmbeddingClient"


def test_cli_probe_algo3_emits_summary_json(monkeypatch, capsys, tmp_path) -> None:
    captured_call: dict[str, object] = {}

    class FakeChatClient:
        def __init__(self, *, api_key: str, model: str) -> None:
            captured_call["api_key"] = api_key
            captured_call["model"] = model

    def fake_run_algo3_probe(*, spec, chat_client) -> dict[str, object]:
        captured_call["run_name"] = spec.run_name
        captured_call["source_labels"] = spec.source_labels
        captured_call["target_labels"] = spec.target_labels
        captured_call["child_count"] = spec.child_count
        captured_call["max_depth"] = spec.max_depth
        captured_call["chat_client_type"] = type(chat_client).__name__
        return {"run_name": spec.run_name, "matched_labels": ["bridge_hit"]}

    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.probe.Algo3ChatClient",
        FakeChatClient,
    )
    monkeypatch.setattr(
        "llm_conceptual_modeling.commands.probe.run_algo3_probe",
        fake_run_algo3_probe,
    )

    exit_code = main(
        [
            "probe",
            "algo3",
            "--run-name",
            "algo3_row0",
            "--model",
            "mistral-small-2603",
            "--source-label",
            "source_a",
            "--target-label",
            "target_a",
            "--target-label",
            "bridge_hit",
            "--child-count",
            "3",
            "--max-depth",
            "2",
            "--output-dir",
            str(tmp_path / "algo3_probe"),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload == {"run_name": "algo3_row0", "matched_labels": ["bridge_hit"]}
    assert captured_call["api_key"] == "test-key"
    assert captured_call["model"] == "mistral-small-2603"
    assert captured_call["source_labels"] == ["source_a"]
    assert captured_call["target_labels"] == ["target_a", "bridge_hit"]
    assert captured_call["child_count"] == 3
    assert captured_call["max_depth"] == 2
    assert captured_call["chat_client_type"] == "FakeChatClient"


def test_cli_probe_requires_mistral_api_key(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    exit_code = main(
        [
            "probe",
            "algo3",
            "--run-name",
            "algo3_row0",
            "--model",
            "mistral-small-2603",
            "--source-label",
            "source_a",
            "--target-label",
            "target_a",
            "--child-count",
            "3",
            "--max-depth",
            "2",
            "--output-dir",
            str(tmp_path / "algo3_probe"),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Missing required environment variable: MISTRAL_API_KEY" in captured.err
