from llm_conceptual_modeling.commands._provider_utils import (
    build_chat_client,
    build_embedding_client,
    resolve_provider_api_key,
)


def test_resolve_provider_api_key_supports_openrouter(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")

    assert resolve_provider_api_key("openrouter") == "openrouter-key"


def test_build_chat_client_uses_openrouter_client(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeOpenRouterChatClient:
        def __init__(self, *, api_key: str, model: str) -> None:
            captured["api_key"] = api_key
            captured["model"] = model

    monkeypatch.setattr(
        "llm_conceptual_modeling.commands._provider_utils.OpenRouterChatClient",
        FakeOpenRouterChatClient,
    )

    client = build_chat_client(
        provider="openrouter",
        api_key="router-key",
        model="gpt-5",
        mistral_chat_client_class=object,
    )

    assert isinstance(client, FakeOpenRouterChatClient)
    assert captured == {"api_key": "router-key", "model": "gpt-5"}


def test_build_embedding_client_uses_openrouter_client(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeOpenRouterEmbeddingClient:
        def __init__(self, *, api_key: str, model: str) -> None:
            captured["api_key"] = api_key
            captured["model"] = model

    monkeypatch.setattr(
        "llm_conceptual_modeling.commands._provider_utils.OpenRouterEmbeddingClient",
        FakeOpenRouterEmbeddingClient,
    )

    client = build_embedding_client(
        provider="openrouter",
        api_key="router-key",
        model="text-embedding-3-large",
        mistral_embedding_client_class=object,
    )

    assert isinstance(client, FakeOpenRouterEmbeddingClient)
    assert captured == {"api_key": "router-key", "model": "text-embedding-3-large"}


def test_resolve_provider_api_key_supports_hf_transformers(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf-token")

    assert resolve_provider_api_key("hf-transformers") == "hf-token"
