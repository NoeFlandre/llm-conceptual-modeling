from __future__ import annotations

import json
import os
from typing import Any, Protocol

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised indirectly in import-light tests
    OpenAI = None  # type: ignore[assignment]

from llm_conceptual_modeling.common.retry import call_with_retry
from llm_conceptual_modeling.common.structured_output import normalize_structured_response

OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterChatCompletionClient(Protocol):
    def complete_json(
        self,
        *,
        prompt: str,
        schema_name: str,
        schema: dict[str, object],
    ) -> dict[str, object]: ...


class OpenRouterChatClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        sdk_client: Any | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url or os.environ.get(
            "OPENROUTER_BASE_URL", OPENROUTER_DEFAULT_BASE_URL
        )
        if sdk_client is not None:
            self._sdk_client = sdk_client
            return
        if OpenAI is None:
            raise ImportError(
                "openai is unavailable; install openai or inject sdk_client for tests"
            )
        self._sdk_client = OpenAI(api_key=api_key, base_url=self._base_url)

    def complete_json(
        self,
        *,
        prompt: str,
        schema_name: str,
        schema: dict[str, object],
    ) -> dict[str, object]:
        response = call_with_retry(
            operation=lambda: self._sdk_client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": schema,
                    },
                },
            ),
            operation_name="openrouter chat completion",
            max_attempts=8,
            initial_delay_seconds=2.0,
            max_delay_seconds=30.0,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenRouter returned an empty chat completion content")
        parsed_content = json.loads(content)
        return normalize_structured_response(parsed_content, schema_name=schema_name)


class OpenRouterEmbeddingClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        sdk_client: Any | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url or os.environ.get(
            "OPENROUTER_BASE_URL", OPENROUTER_DEFAULT_BASE_URL
        )
        if sdk_client is not None:
            self._sdk_client = sdk_client
            return
        if OpenAI is None:
            raise ImportError(
                "openai is unavailable; install openai or inject sdk_client for tests"
            )
        self._sdk_client = OpenAI(api_key=api_key, base_url=self._base_url)

    def embed_texts(self, texts: list[str]) -> dict[str, list[float]]:
        response = call_with_retry(
            operation=lambda: self._sdk_client.embeddings.create(
                model=self._model,
                input=texts,
            ),
            operation_name="openrouter embeddings",
            max_attempts=8,
            initial_delay_seconds=2.0,
            max_delay_seconds=30.0,
        )
        data_items = response.data
        embeddings_by_label: dict[str, list[float]] = {}

        for text, item in zip(texts, data_items, strict=True):
            embeddings_by_label[text] = list(item.embedding)

        return embeddings_by_label
