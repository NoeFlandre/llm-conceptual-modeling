from urllib.error import URLError

from llm_conceptual_modeling.algo2.embeddings import (
    MistralEmbeddingClient,
    build_embeddings_by_label,
    compute_average_best_match_similarity,
)


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> dict[str, list[float]]:
        self.calls.append(texts)
        return {text: [float(index), 1.0] for index, text in enumerate(texts, start=1)}


def test_build_embeddings_by_label_preserves_input_order_for_missing_duplicates() -> None:
    client = FakeEmbeddingClient()

    actual = build_embeddings_by_label(
        labels=["alpha", "beta", "alpha"],
        client=client,
    )

    assert actual == {
        "alpha": [1.0, 1.0],
        "beta": [2.0, 1.0],
    }
    assert client.calls == [["alpha", "beta"]]


def test_mistral_embedding_client_builds_expected_request_payload() -> None:
    captured_request: dict[str, object] = {}

    def fake_post_json(*, url: str, api_key: str, payload: dict[str, object]) -> dict[str, object]:
        captured_request["url"] = url
        captured_request["api_key"] = api_key
        captured_request["payload"] = payload
        return {
            "data": [
                {"embedding": [0.1, 0.2]},
                {"embedding": [0.3, 0.4]},
            ]
        }

    client = MistralEmbeddingClient(
        api_key="test-key",
        model="mistral-embed-2312",
        post_json=fake_post_json,
    )

    actual = client.embed_texts(["alpha", "beta"])

    assert actual == {
        "alpha": [0.1, 0.2],
        "beta": [0.3, 0.4],
    }
    assert captured_request["url"] == "https://api.mistral.ai/v1/embeddings"
    assert captured_request["api_key"] == "test-key"
    assert captured_request["payload"] == {
        "model": "mistral-embed-2312",
        "input": ["alpha", "beta"],
    }


def test_mistral_embedding_client_retries_transient_transport_errors() -> None:
    calls = {"count": 0}

    def flaky_post_json(*, url: str, api_key: str, payload: dict[str, object]) -> dict[str, object]:
        calls["count"] += 1
        if calls["count"] < 3:
            raise URLError("temporary network issue")
        return {
            "data": [
                {"embedding": [0.1, 0.2]},
                {"embedding": [0.3, 0.4]},
            ]
        }

    client = MistralEmbeddingClient(
        api_key="test-key",
        model="mistral-embed-2312",
        post_json=flaky_post_json,
    )

    actual = client.embed_texts(["alpha", "beta"])

    assert actual == {
        "alpha": [0.1, 0.2],
        "beta": [0.3, 0.4],
    }
    assert calls["count"] == 3


def test_compute_average_best_match_similarity_uses_embedding_client_once() -> None:
    client = FakeEmbeddingClient()
    expected_similarity = (0.8944271909999159 + 0.9899494936611665) / 2

    actual = compute_average_best_match_similarity(
        candidate_labels=["alpha", "beta"],
        seed_labels=["gamma"],
        client=client,
    )

    assert round(actual, 6) == round(expected_similarity, 6)
    assert client.calls == [["alpha", "beta", "gamma"]]
