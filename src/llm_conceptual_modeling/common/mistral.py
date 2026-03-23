"""Shared Mistral chat-client utilities.

This module contains the code that was duplicated across
algo1/, algo2/, and algo3/ mistral.py files:

- :class:`MistralChatClient` — thin SDK wrapper with JSON-schema completion
- :class:`ChatCompletionClient` — protocol for any chat client implementing
  ``complete_json``
- ``Edge`` — type alias for ``tuple[str, str]``
- Knowledge-map prompt formatting helpers used by Method 1 and Method 2

Each algorithm re-exports its own prompt-builder functions from its own
mistral.py while delegating the shared primitives to this module.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Protocol

try:
    from mistralai.client import Mistral
except ImportError:  # pragma: no cover - exercised indirectly in import-light tests
    Mistral = None  # type: ignore[assignment]

from llm_conceptual_modeling.common.retry import call_with_retry

if TYPE_CHECKING:
    pass

__all__ = [
    "ChatCompletionClient",
    "Edge",
    "MistralChatClient",
    "_build_adjacency_matrix",
    "_build_notation_section",
    "_collect_ordered_nodes",
    "_format_knowledge_map",
    "_format_knowledge_map_as_adjacency",
    "_format_knowledge_map_as_edge_list",
]


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Edge = tuple[str, str]


# ---------------------------------------------------------------------------
# Mistral SDK client
# ---------------------------------------------------------------------------


class MistralChatClient:
    """Thin Mistral SDK wrapper that wraps chat completions with JSON parsing.

    Parameters
    ----------
    api_key:
        Mistral API key. Ignored when ``sdk_client`` is supplied directly.
    model:
        Model name sent to the Mistral API (e.g. ``"mistral-small-latest"``).
    sdk_client:
        Optional pre-constructed Mistral SDK client. When omitted the client
        is created from ``api_key``. Useful for testing with a fake client.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        sdk_client: Any | None = None,
    ) -> None:
        self._model = model
        if sdk_client is not None:
            self._sdk_client = sdk_client
            return
        if Mistral is None:
            raise ImportError(
                "mistralai.client.Mistral is unavailable; install mistralai or "
                "inject sdk_client for tests"
            )
        self._sdk_client = Mistral(api_key=api_key)

    def complete_json(
        self,
        *,
        prompt: str,
        schema_name: str,
        schema: dict[str, object],
    ) -> dict[str, object]:
        """Call the Mistral chat API and return the parsed JSON response.

        Parameters
        ----------
        prompt:
            User-message content sent to the model.
        schema_name:
            Name of the JSON schema used for structured completion.
        schema:
            JSON schema definition forwarded to the Mistral API.

        Returns
        -------
        dict[str, object]
            Parsed response body.

        Raises
        ------
        ValueError
            When the model returns a response with empty (None) content.
        """
        response = call_with_retry(
            operation=lambda: self._sdk_client.chat.complete(
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
            operation_name="mistral chat completion",
            max_attempts=8,
            initial_delay_seconds=2.0,
            max_delay_seconds=30.0,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Mistral returned an empty chat completion content")
        parsed_content = json.loads(content)
        return parsed_content


# ---------------------------------------------------------------------------
# ChatCompletionClient protocol
# ---------------------------------------------------------------------------


class ChatCompletionClient(Protocol):
    """Minimal protocol for a chat client that returns structured JSON.

    Both :class:`MistralChatClient` and the fake clients used in tests
    satisfy this protocol.
    """

    def complete_json(
        self,
        *,
        prompt: str,
        schema_name: str,
        schema: dict[str, object],
    ) -> dict[str, object]: ...


# ---------------------------------------------------------------------------
# Knowledge-map prompt formatting helpers
#
# These helpers are shared between Method 1 (algo1/) and Method 2 (algo3/).
# Method 3 (algo2/) does not use knowledge maps and therefore does not need
# these helpers.
# ---------------------------------------------------------------------------


def _build_notation_section(
    *,
    use_adjacency_notation: bool,
    use_array_representation: bool,
) -> str:
    if use_adjacency_notation:
        if use_array_representation:
            return (
                "The knowledge map is encoded using a list of nodes and an associated "
                "adjacency matrix. The adjacency matrix is an n*n square matrix that "
                "represents whether each edge exists."
            )

        return (
            "The knowledge map is encoded using tags for nodes and an associated adjacency matrix."
        )

    if use_array_representation:
        return "The knowledge map is encoded using arrays of directed edges."

    return "The knowledge map is encoded using tags that store directed edges."


def _build_example_section() -> str:
    return (
        "Here is an example of a desired output for your task. "
        "We have the list of concepts ['capacity to hire', 'bad employees', "
        "'good reputation']. In this example, you could recommend these 9 new concepts: "
        "'employment potential', 'hiring capability', 'staffing ability', "
        "'underperformers', 'inefficient staff', 'problematic workers', "
        "'positive image', 'favorable standing', 'high regard'."
    )


def _build_counterexample_section() -> str:
    return (
        "Here is an example of a bad output that we do not want to see. "
        "A bad output would propose unrelated concepts such as 'moon', 'dog', "
        "or 'thermodynamics'."
    )


def _format_knowledge_map(
    edges: list[Edge],
    *,
    prompt_config: Any,  # duck-typed: attrs use_adjacency_notation + use_array_representation
) -> str:
    if prompt_config.use_adjacency_notation:
        return _format_knowledge_map_as_adjacency(
            edges,
            use_array_representation=prompt_config.use_array_representation,
        )

    return _format_knowledge_map_as_edge_list(
        edges,
        use_array_representation=prompt_config.use_array_representation,
    )


def _format_knowledge_map_as_adjacency(
    edges: list[Edge],
    *,
    use_array_representation: bool,
) -> str:
    ordered_nodes = _collect_ordered_nodes(edges)
    adjacency_matrix = _build_adjacency_matrix(
        edges,
        ordered_nodes=ordered_nodes,
    )

    if use_array_representation:
        return str(
            {
                "nodes": ordered_nodes,
                "adjacency_matrix": adjacency_matrix,
            }
        )

    node_tags = "".join(f"<node>{node}</node>" for node in ordered_nodes)
    return (
        "<knowledge-map>"
        f"<nodes>{node_tags}</nodes>"
        f"<adjacency-matrix>{adjacency_matrix}</adjacency-matrix>"
        "</knowledge-map>"
    )


def _format_knowledge_map_as_edge_list(
    edges: list[Edge],
    *,
    use_array_representation: bool,
) -> str:
    if use_array_representation:
        return str(edges)

    edge_tags: list[str] = []

    for source, target in edges:
        edge_tag = f"<edge source='{source}' target='{target}' />"
        edge_tags.append(edge_tag)

    joined_edge_tags = "".join(edge_tags)
    return f"<knowledge-map>{joined_edge_tags}</knowledge-map>"


def _collect_ordered_nodes(edges: list[Edge]) -> list[str]:
    ordered_nodes: list[str] = []
    seen_nodes: set[str] = set()

    for source, target in edges:
        for node in (source, target):
            if node in seen_nodes:
                continue
            ordered_nodes.append(node)
            seen_nodes.add(node)

    return ordered_nodes


def _build_adjacency_matrix(
    edges: list[Edge],
    *,
    ordered_nodes: list[str],
) -> list[list[int]]:
    node_positions: dict[str, int] = {}

    for index, node in enumerate(ordered_nodes):
        node_positions[node] = index

    matrix_size = len(ordered_nodes)
    adjacency_matrix = [[0 for _ in range(matrix_size)] for _ in range(matrix_size)]

    for source, target in edges:
        source_index = node_positions[source]
        target_index = node_positions[target]
        adjacency_matrix[source_index][target_index] = 1

    return adjacency_matrix
