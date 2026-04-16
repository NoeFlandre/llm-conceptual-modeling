from llm_conceptual_modeling.common import anthropic, mistral, openrouter
from llm_conceptual_modeling.common.client_protocols import (
    ChatCompletionClient,
    EmbeddingClient,
)


def test_chat_completion_client_has_one_canonical_definition() -> None:
    assert mistral.ChatCompletionClient is ChatCompletionClient
    assert anthropic.ChatCompletionClient is ChatCompletionClient
    assert openrouter.ChatCompletionClient is ChatCompletionClient
    assert ChatCompletionClient.__module__ == "llm_conceptual_modeling.common.client_protocols"


def test_embedding_client_has_one_canonical_definition() -> None:
    from llm_conceptual_modeling.algo2.embeddings import EmbeddingClient as Algo2EmbeddingClient

    assert Algo2EmbeddingClient is EmbeddingClient
    assert EmbeddingClient.__module__ == "llm_conceptual_modeling.common.client_protocols"
