"""Anthropic API client for MiniMax-M2.7 and other Anthropic models.

NOTE: This client is for debugging and research purposes only.
MiniMax-M2.7 support via the Anthropic API is experimental and not
part of the paper's primary evaluation pipeline, which uses Mistral
models via the official Mistral SDK.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import time
from collections.abc import Mapping
from typing import Any, cast

import anthropic
import httpx

from llm_conceptual_modeling.common.client_protocols import (
    ChatCompletionClient as _ChatCompletionClient,
)

logger = logging.getLogger(__name__)

ChatCompletionClient = _ChatCompletionClient


def _call_with_retry(
    operation: Any,
    operation_name: str,
    max_attempts: int = 8,
    initial_delay_seconds: float = 2.0,
    backoff_factor: float = 2.0,
    max_delay_seconds: float = 30.0,
) -> Any:
    """Retry wrapper for Anthropic API calls with exponential backoff."""
    attempt = 1
    delay_seconds = initial_delay_seconds

    while True:
        try:
            return operation()
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            retryable = True
            exception = exc
        except httpx.HTTPStatusError as exc:
            # Retry on rate limiting and server errors
            retryable = exc.response.status_code in (429, 500, 502, 503, 504)
            exception = exc
        except anthropic.APIError as exc:
            # Anthropic API errors - retry on rate limiting
            retryable = getattr(exc, "status_code", 0) in (429, 500, 502, 503, 504)
            exception = exc

        if not retryable or attempt >= max_attempts:
            logger.error(
                "Exhausted retries; operation=%s attempts=%s/%s error_type=%s",
                operation_name,
                attempt,
                max_attempts,
                type(exception).__name__,
            )
            raise exception

        logger.warning(
            "Retrying; operation=%s attempt=%s/%s delay_seconds=%.3f",
            operation_name,
            attempt + 1,
            max_attempts,
            delay_seconds,
        )
        time.sleep(delay_seconds)
        attempt += 1
        delay_seconds = min(delay_seconds * backoff_factor, max_delay_seconds)


class AnthropicChatClient:
    """Chat completion client using the Anthropic API.

    NOTE: For debugging and research purposes only. MiniMax-M2.7 support
    via the Anthropic API is experimental and not part of the paper's
    primary evaluation pipeline.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_tokens: int | None = None,
        sdk_client: Any | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        env_max_tokens = os.environ.get("ANTHROPIC_MAX_TOKENS")
        if max_tokens is not None:
            self._max_tokens = max_tokens
        elif env_max_tokens is not None:
            self._max_tokens = int(env_max_tokens)
        else:
            self._max_tokens = 196608
        base_url = os.environ.get("ANTHROPIC_BASE_URL")
        if sdk_client is not None:
            self._sdk_client = sdk_client
        elif base_url:
            self._sdk_client = anthropic.Anthropic(auth_token=api_key, base_url=base_url)
        else:
            self._sdk_client = anthropic.Anthropic(api_key=api_key)

    def complete_json(
        self,
        *,
        prompt: str,
        schema_name: str,
        schema: dict[str, object],
        temperature: float = 0.0,
    ) -> dict[str, object]:
        """Call the Anthropic API with JSON output and return parsed result."""

        def operation() -> Any:
            return self._sdk_client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system="You are a helpful assistant.",
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }
                ],
                temperature=temperature,
            )

        response = _call_with_retry(
            operation=operation,
            operation_name="anthropic chat completion",
            max_attempts=8,
            initial_delay_seconds=2.0,
            max_delay_seconds=30.0,
        )

        # Extract text from content blocks
        # MiniMax may return thinking blocks first, then text blocks
        text_parts: list[str] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)

        if not text_parts:
            raise ValueError("Anthropic returned no text content")

        # Join all text parts and parse as JSON
        full_text = "".join(text_parts)
        try:
            parsed = json.loads(full_text)
        except json.JSONDecodeError:
            parsed = self._recover_non_json_response(full_text=full_text, schema_name=schema_name)

        return parsed

    def _recover_non_json_response(
        self,
        *,
        full_text: str,
        schema_name: str,
    ) -> dict[str, object]:
        parsed: object | None = None
        stripped = full_text.strip()

        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            parsed = self._parse_tuple_list_text(stripped)

        if schema_name == "edge_list":
            return {"edges": self._coerce_edge_list(parsed)}
        if schema_name == "vote_list":
            return {"votes": self._coerce_string_list(parsed)}
        if schema_name == "label_list":
            return {"labels": self._coerce_string_list(parsed)}
        if isinstance(parsed, dict):
            return cast(dict[str, object], parsed)

        raise ValueError(f"Anthropic response was not valid JSON: Response: {full_text[:500]}")

    @staticmethod
    def _parse_tuple_list_text(text: str) -> object:
        tuple_texts = re.findall(r"\(([^()]*)\)", text)
        if not tuple_texts:
            raise ValueError("No tuple-like content found")
        parsed_tuples: list[tuple[str, str]] = []
        for tuple_text in tuple_texts:
            parts = [part.strip().strip("'\"") for part in tuple_text.split(",", 1)]
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(f"Could not parse tuple content: {tuple_text}")
            parsed_tuples.append((parts[0], parts[1]))
        return parsed_tuples

    @staticmethod
    def _coerce_edge_list(parsed: object) -> list[dict[str, str]]:
        if isinstance(parsed, Mapping):
            parsed_mapping = cast(Mapping[str, object], parsed)
            edges = parsed_mapping.get("edges")
            if isinstance(edges, list):
                coerced_edges: list[dict[str, str]] = []
                for edge in edges:
                    if isinstance(edge, Mapping) and "source" in edge and "target" in edge:
                        edge_mapping = cast(Mapping[str, object], edge)
                        coerced_edges.append(
                            {
                                "source": str(edge_mapping["source"]),
                                "target": str(edge_mapping["target"]),
                            }
                        )
                return coerced_edges
            raise ValueError("Parsed response does not contain an edges list")

        if not isinstance(parsed, list):
            raise ValueError("Parsed response is not a list")

        coerced_edges: list[dict[str, str]] = []
        for item in parsed:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                raise ValueError(f"Unsupported edge item: {item!r}")
            coerced_edges.append({"source": str(item[0]), "target": str(item[1])})
        return coerced_edges

    @staticmethod
    def _coerce_string_list(parsed: object) -> list[str]:
        if isinstance(parsed, dict):
            first_value = next(iter(parsed.values()), [])
            if not isinstance(first_value, list):
                raise ValueError("Parsed response does not contain a list value")
            return [str(item) for item in first_value]

        if not isinstance(parsed, list):
            raise ValueError("Parsed response is not a list")

        return [str(item) for item in parsed]
