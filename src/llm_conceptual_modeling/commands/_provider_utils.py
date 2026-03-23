import os
from typing import Any


def resolve_provider_api_key(provider: str) -> str:
    environment_variable = (
        "ANTHROPIC_API_KEY" if provider == "anthropic" else "MISTRAL_API_KEY"
    )
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
    return mistral_chat_client_class(api_key=api_key, model=model)
