import json
from typing import Protocol
from urllib import request

from llm_conceptual_modeling.algo2.expansion import average_best_match_similarity


class EmbeddingClient(Protocol):
    def embed_texts(self, texts: list[str]) -> dict[str, list[float]]: ...


class MistralEmbeddingClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        post_json: "PostJsonFunction" | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._post_json = post_json or _post_json

    def embed_texts(self, texts: list[str]) -> dict[str, list[float]]:
        payload = {
            "model": self._model,
            "input": texts,
        }
        response = self._post_json(
            url="https://api.mistral.ai/v1/embeddings",
            api_key=self._api_key,
            payload=payload,
        )
        data_items = response["data"]
        embeddings_by_label: dict[str, list[float]] = {}

        for text, item in zip(texts, data_items, strict=True):
            embedding = item["embedding"]
            embeddings_by_label[text] = embedding

        return embeddings_by_label


class PostJsonFunction(Protocol):
    def __call__(
        self,
        *,
        url: str,
        api_key: str,
        payload: dict[str, object],
    ) -> dict[str, object]: ...


def build_embeddings_by_label(
    *,
    labels: list[str],
    client: EmbeddingClient,
) -> dict[str, list[float]]:
    unique_labels: list[str] = []
    seen_labels: set[str] = set()

    for label in labels:
        already_seen = label in seen_labels
        if already_seen:
            continue
        unique_labels.append(label)
        seen_labels.add(label)

    embeddings_by_label = client.embed_texts(unique_labels)
    return embeddings_by_label


def compute_average_best_match_similarity(
    *,
    candidate_labels: list[str],
    seed_labels: list[str],
    client: EmbeddingClient,
) -> float:
    labels_to_embed = candidate_labels + seed_labels
    embeddings_by_label = build_embeddings_by_label(
        labels=labels_to_embed,
        client=client,
    )
    similarity = average_best_match_similarity(
        candidate_labels=candidate_labels,
        seed_labels=seed_labels,
        embeddings_by_label=embeddings_by_label,
    )
    return similarity


def _post_json(
    *,
    url: str,
    api_key: str,
    payload: dict[str, object],
) -> dict[str, object]:
    body_text = json.dumps(payload)
    body_bytes = body_text.encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    http_request = request.Request(
        url=url,
        data=body_bytes,
        headers=headers,
        method="POST",
    )
    with request.urlopen(http_request) as response:
        response_bytes = response.read()
    response_text = response_bytes.decode("utf-8")
    parsed_response = json.loads(response_text)
    return parsed_response
