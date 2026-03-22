import typing
"""Tests for AnthropicChatClient (MiniMax-M2.7 support)."""

from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from llm_conceptual_modeling.common.anthropic import AnthropicChatClient


def _fake_anthropic_response(content: str) -> Any:
    """Create a fake Anthropic response with text content."""
    # Anthropic returns a Message object with content blocks
    text_block = SimpleNamespace(type="text", text=content)
    return SimpleNamespace(
        content=[text_block],
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=20,
        ),
    )


class TestAnthropicChatClient:
    def test_complete_json_returns_parsed_dict(self) -> None:
        """Basic test: verify complete_json returns a dict from JSON text response."""

        class FakeMessages:
            def create(self, **kwargs: Any) -> Any:
                # Return a simple JSON response
                return _fake_anthropic_response('{"edges": [{"source": "a", "target": "b"}]}')

        class FakeAnthropic:
            def __init__(self, **kwargs: Any) -> None:
                self.messages = FakeMessages()

        client = AnthropicChatClient(
            api_key="test-key",
            model="MiniMax-M2.7",
            sdk_client=FakeAnthropic(),
        )

        result = client.complete_json(
            prompt="generate edges",
            schema_name="edge_list",
            schema={
                "type": "object",
                "properties": {
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string"},
                                "target": {"type": "string"},
                            },
                            "required": ["source", "target"],
                        },
                    }
                },
                "required": ["edges"],
            },
        )

        assert result == {"edges": [{"source": "a", "target": "b"}]}

    def test_complete_json_passes_required_fields_to_api(self) -> None:
        """Verify the Anthropic API receives model, messages, and JSON config."""

        captured: dict[str, Any] = {}

        class FakeMessages:
            def create(self, **kwargs: Any) -> Any:
                captured.update(kwargs)
                return _fake_anthropic_response('{"edges": []}')

        class FakeAnthropic:
            def __init__(self, **kwargs: Any) -> None:
                self.messages = FakeMessages()

        client = AnthropicChatClient(
            api_key="test-key",
            model="MiniMax-M2.7",
            sdk_client=FakeAnthropic(),
        )

        client.complete_json(
            prompt="test prompt",
            schema_name="edge_list",
            schema={"type": "object"},
        )

        assert captured["model"] == "MiniMax-M2.7"
        assert captured["max_tokens"] == 4096
        assert len(captured["messages"]) == 1
        assert captured["messages"][0]["role"] == "user"

    def test_complete_json_with_custom_temperature(self) -> None:
        """Temperature parameter is forwarded to the API."""

        captured: dict[str, Any] = {}

        class FakeMessages:
            def create(self, **kwargs: Any) -> Any:
                captured.update(kwargs)
                return _fake_anthropic_response('{"edges": []}')

        class FakeAnthropic:
            def __init__(self, **kwargs: Any) -> None:
                self.messages = FakeMessages()

        client = AnthropicChatClient(
            api_key="test-key",
            model="MiniMax-M2.7",
            sdk_client=FakeAnthropic(),
        )

        client.complete_json(
            prompt="test",
            schema_name="edge_list",
            schema={"type": "object"},
            temperature=0.7,
        )

        assert captured["temperature"] == 0.7

    def test_complete_json_retry_on_transient_error(self) -> None:
        """Retries on transient httpx errors, succeeds after failures."""

        call_count = {"n": 0}

        class FakeMessages:
            def create(self, **kwargs: Any) -> Any:
                call_count["n"] += 1
                if call_count["n"] < 3:
                    raise httpx.ConnectError("temporary network issue")
                return _fake_anthropic_response('{"edges": [{"source": "x", "target": "y"}]}')

        class FakeAnthropic:
            def __init__(self, **kwargs: Any) -> None:
                self.messages = FakeMessages()

        client = AnthropicChatClient(
            api_key="test-key",
            model="MiniMax-M2.7",
            sdk_client=FakeAnthropic(),
        )

        result = client.complete_json(
            prompt="test",
            schema_name="edge_list",
            schema={"type": "object"},
        )

        assert result == {"edges": [{"source": "x", "target": "y"}]}
        assert call_count["n"] == 3

    def test_complete_json_raises_non_retryable_error(self) -> None:
        """HTTP 400 (bad request) is not retried."""

        class FakeMessages:
            def create(self, **kwargs: Any) -> Any:
                raise httpx.HTTPStatusError(
                    "bad request",
                    request=typing.cast(httpx.Request, SimpleNamespace(url="http://example.com")),
                    response=typing.cast(httpx.Response, SimpleNamespace(status_code=400)),
                )

        class FakeAnthropic:
            def __init__(self, **kwargs: Any) -> None:
                self.messages = FakeMessages()

        client = AnthropicChatClient(
            api_key="test-key",
            model="MiniMax-M2.7",
            sdk_client=FakeAnthropic(),
        )

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            client.complete_json(
                prompt="test",
                schema_name="edge_list",
                schema={"type": "object"},
            )

        assert exc_info.value.response.status_code == 400
