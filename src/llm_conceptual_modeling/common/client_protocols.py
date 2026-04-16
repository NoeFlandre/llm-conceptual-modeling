from typing import Protocol


class ChatCompletionClient(Protocol):
    def complete_json(
        self,
        *,
        prompt: str,
        schema_name: str,
        schema: dict[str, object],
    ) -> dict[str, object]: ...


class EmbeddingClient(Protocol):
    def embed_texts(self, texts: list[str]) -> dict[str, list[float]]: ...
