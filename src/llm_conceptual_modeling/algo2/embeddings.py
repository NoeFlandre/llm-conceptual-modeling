from typing import Any, Protocol

from mistralai.client import Mistral

from llm_conceptual_modeling.algo2.expansion import average_best_match_similarity
from llm_conceptual_modeling.common.retry import call_with_retry


class EmbeddingClient(Protocol):
    def embed_texts(self, texts: list[str]) -> dict[str, list[float]]: ...


class MistralEmbeddingClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        sdk_client: Any | None = None,
    ) -> None:
        self._model = model
        self._sdk_client = sdk_client or Mistral(api_key=api_key)

    def embed_texts(self, texts: list[str]) -> dict[str, list[float]]:
        response = call_with_retry(
            operation=lambda: self._sdk_client.embeddings.create(
                model=self._model,
                inputs=texts,
            ),
            operation_name="mistral embeddings",
        )
        data_items = response.data
        embeddings_by_label: dict[str, list[float]] = {}

        for text, item in zip(texts, data_items, strict=True):
            embeddings_by_label[text] = list(item.embedding)

        return embeddings_by_label


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
