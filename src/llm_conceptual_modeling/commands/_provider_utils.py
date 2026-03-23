import os
from typing import Any

OpenRouterChatClient = None
OpenRouterEmbeddingClient = None


def resolve_provider_api_key(provider: str) -> str:
    environment_variables = {
        "anthropic": "ANTHROPIC_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    if provider not in environment_variables:
        raise ValueError(f"Unsupported provider: {provider}")

    environment_variable = environment_variables[provider]
    api_key = os.environ.get(environment_variable)
    if api_key:
        return api_key

    raise ValueError(f"Missing required environment variable: {environment_variable}")


def build_chat_client(
    *,
    provider: str,
    api_key: str,
    model: str,
    mistral_chat_client_class: Any,
) -> Any:
    if provider == "anthropic":
        from llm_conceptual_modeling.common.anthropic import AnthropicChatClient

        return AnthropicChatClient(api_key=api_key, model=model)
    if provider == "openrouter":
        global OpenRouterChatClient
        if OpenRouterChatClient is None:
            from llm_conceptual_modeling.common.openrouter import (
                OpenRouterChatClient as loaded_openrouter_chat_client,
            )

            OpenRouterChatClient = loaded_openrouter_chat_client
        return OpenRouterChatClient(api_key=api_key, model=model)
    return mistral_chat_client_class(api_key=api_key, model=model)


def build_embedding_client(
    *,
    provider: str,
    api_key: str,
    model: str,
    mistral_embedding_client_class: Any,
) -> Any:
    if provider == "openrouter":
        global OpenRouterEmbeddingClient
        if OpenRouterEmbeddingClient is None:
            from llm_conceptual_modeling.common.openrouter import (
                OpenRouterEmbeddingClient as loaded_openrouter_embedding_client,
            )

            OpenRouterEmbeddingClient = loaded_openrouter_embedding_client
        return OpenRouterEmbeddingClient(api_key=api_key, model=model)
    return mistral_embedding_client_class(api_key=api_key, model=model)
