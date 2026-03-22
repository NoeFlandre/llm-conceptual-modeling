"""Anthropic API client for MiniMax-M2.7 and other Anthropic models.

NOTE: This client is for debugging and research purposes only.
MiniMax-M2.7 support via the Anthropic API is experimental and not
part of the paper's primary evaluation pipeline, which uses Mistral
models via the official Mistral SDK.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Protocol

import anthropic
import httpx

logger = logging.getLogger(__name__)


class ChatCompletionClient(Protocol):
    """Protocol defining the chat completion interface used across algorithm modules."""

    def complete_json(
        self,
        *,
        prompt: str,
        schema_name: str,
        schema: dict[str, object],
    ) -> dict[str, object]:
        ...


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
        sdk_client: Any | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        base_url = os.environ.get("ANTHROPIC_BASE_URL")
        if sdk_client is not None:
            self._sdk_client = sdk_client
        elif base_url:
            self._sdk_client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
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
        del schema_name  # unused - schema guides the prompt but isn't sent as a name

        def operation() -> Any:
            return self._sdk_client.messages.create(
                model=self._model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }
                ],
                temperature=temperature,
                thinking={"type": "disabled"},
                output_config={"type": "object"},
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
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Anthropic response was not valid JSON: {exc}. Response: {full_text[:500]}"
            ) from exc

        return parsed
