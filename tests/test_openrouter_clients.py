import json

from llm_conceptual_modeling.common.openrouter import (
    OpenRouterChatClient,
    OpenRouterEmbeddingClient,
)


class FakeChatResponse:
    def __init__(self, content: str) -> None:
        self.choices = [
            type(
                "Choice",
                (),
                {
                    "message": type("Message", (), {"content": content})(),
                },
            )()
        ]


class FakeEmbeddingResponse:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.data = [type("Item", (), {"embedding": vector})() for vector in vectors]


def test_openrouter_chat_client_uses_openai_compatible_chat_api() -> None:
    captured: dict[str, object] = {}

    class FakeChatCompletions:
        def create(self, **kwargs: object) -> FakeChatResponse:
            captured.update(kwargs)
            return FakeChatResponse(json.dumps({"labels": ["bridge_a"]}))

    class FakeSDKClient:
        def __init__(self) -> None:
            self.chat = type("Chat", (), {"completions": FakeChatCompletions()})()

    client = OpenRouterChatClient(
        api_key="router-key",
        model="gpt-5",
        sdk_client=FakeSDKClient(),
    )

    actual = client.complete_json(
        prompt="hello",
        schema_name="label_list",
        schema={"type": "object"},
    )

    assert actual == {"labels": ["bridge_a"]}
    assert captured["model"] == "gpt-5"
    assert captured["temperature"] == 0.0
    assert captured["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["response_format"]["json_schema"]["name"] == "label_list"


def test_openrouter_embedding_client_uses_openai_compatible_embeddings_api() -> None:
    captured: dict[str, object] = {}

    class FakeEmbeddings:
        def create(self, **kwargs: object) -> FakeEmbeddingResponse:
            captured.update(kwargs)
            return FakeEmbeddingResponse([[0.1, 0.2], [0.3, 0.4]])

    class FakeSDKClient:
        def __init__(self) -> None:
            self.embeddings = FakeEmbeddings()

    client = OpenRouterEmbeddingClient(
        api_key="router-key",
        model="text-embedding-3-large",
        sdk_client=FakeSDKClient(),
    )

    actual = client.embed_texts(["alpha", "beta"])

    assert actual == {
        "alpha": [0.1, 0.2],
        "beta": [0.3, 0.4],
    }
    assert captured == {
        "model": "text-embedding-3-large",
        "input": ["alpha", "beta"],
    }
