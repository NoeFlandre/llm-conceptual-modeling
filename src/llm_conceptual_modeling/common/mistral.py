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

import ast
import importlib
import json
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

try:
    from pydantic import BaseModel
except ImportError:  # pragma: no cover - pydantic is an explicit runtime dependency
    BaseModel = None  # type: ignore[assignment]

from llm_conceptual_modeling.common.client_protocols import (
    ChatCompletionClient as _ChatCompletionClient,
)
from llm_conceptual_modeling.common.retry import call_with_retry
from llm_conceptual_modeling.common.structured_output import normalize_structured_response
from llm_conceptual_modeling.common.types import Edge

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


ChatCompletionClient = _ChatCompletionClient


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
        mistral_client_class = _resolve_mistral_client_class()
        if mistral_client_class is None:
            raise ImportError(
                "mistralai.client.Mistral is unavailable; install mistralai or "
                "inject sdk_client for tests"
            )
        self._sdk_client = mistral_client_class(api_key=api_key)

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
            operation=lambda: _complete_structured_json(
                sdk_client=self._sdk_client,
                model=self._model,
                prompt=prompt,
                schema_name=schema_name,
                schema=schema,
            ),
            operation_name="mistral chat completion",
            max_attempts=8,
            initial_delay_seconds=2.0,
            max_delay_seconds=30.0,
        )
        return response


class _EdgeItemModel(BaseModel):
    source: str
    target: str


class _EdgeListModel(BaseModel):
    edges: list[_EdgeItemModel]


class _VoteListModel(BaseModel):
    votes: list[str]


class _LabelListModel(BaseModel):
    labels: list[str]


_STRUCTURED_RESPONSE_MODELS: dict[str, type[BaseModel]] = {
    "edge_list": _EdgeListModel,
    "vote_list": _VoteListModel,
    "label_list": _LabelListModel,
}

_normalize_structured_response = normalize_structured_response


def _resolve_mistral_client_class() -> Any | None:
    try:
        module = importlib.import_module("mistralai")
        client_class = getattr(module, "Mistral", None)
        if client_class is not None:
            return client_class
    except ImportError:
        pass

    try:
        module = importlib.import_module("mistralai.client")
        return getattr(module, "Mistral", None)
    except ImportError:
        return None


def _complete_structured_json(
    *,
    sdk_client: Any,
    model: str,
    prompt: str,
    schema_name: str,
    schema: dict[str, object],
) -> dict[str, object]:
    parse_model = _STRUCTURED_RESPONSE_MODELS.get(schema_name)
    if parse_model is not None and hasattr(sdk_client.chat, "parse"):
        parsed_response = sdk_client.chat.parse(
            response_format=parse_model,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        parsed_choice = parsed_response.choices[0]
        parsed_message = parsed_choice.message
        if parsed_message is None or parsed_message.parsed is None:
            raise ValueError(f"Mistral returned an empty parsed completion for {schema_name}")
        parsed_payload = parsed_message.parsed.model_dump()
        return _normalize_structured_response(parsed_payload, schema_name=schema_name)

    response = sdk_client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema,
            },
        },
    )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("Mistral returned an empty chat completion content")
    parsed_content: object
    try:
        parsed_content = json.loads(content)
    except json.JSONDecodeError:
        parsed_content = _recover_non_json_response(content=content, schema_name=schema_name)
    return normalize_structured_response(parsed_content, schema_name=schema_name)


def _recover_non_json_response(*, content: str, schema_name: str) -> object:
    stripped = content.strip()

    if schema_name == "vote_list":
        token_matches = re.findall(r"\b[YyNn]\b", stripped)
        if token_matches:
            return _coerce_string_list([token.upper() for token in token_matches])

    try:
        parsed: object = ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        parsed = _parse_tuple_list_text(stripped)

    if schema_name == "edge_list":
        return _coerce_edge_list(parsed)
    if schema_name == "vote_list":
        return _coerce_string_list(parsed)
    if schema_name == "label_list":
        return _coerce_string_list(parsed)
    return parsed


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


def _coerce_string_list(parsed: object) -> list[str]:
    if isinstance(parsed, dict):
        first_value = next(iter(parsed.values()), [])
        if not isinstance(first_value, list):
            raise ValueError("Parsed response does not contain a list value")
        return [str(item) for item in first_value]

    if not isinstance(parsed, list):
        raise ValueError("Parsed response is not a list")

    return [str(item) for item in parsed]


# ---------------------------------------------------------------------------
# Knowledge-map prompt formatting helpers
#
# These helpers are shared between Method 1 (algo1/) and Method 2 (algo2/).
# Method 3 (algo3/) does not use knowledge maps and therefore does not need
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
            "The knowledge map is encoded using a tag-array representation. The map is wrapped "
            "in <ARRAY><NODES>...</NODES></ARRAY>. Each <NODE> stores its ID and only the "
            "outgoing targets that are directly connected to it."
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
        return (
            f"the list of nodes {ordered_nodes} and the associated adjacency matrix "
            f"{adjacency_matrix}"
        )

    edge_targets_by_source: dict[str, list[str]] = {node: [] for node in ordered_nodes}
    for source_node, target_node in edges:
        edge_targets_by_source[source_node].append(target_node)

    node_sections: list[str] = []
    for source_node in ordered_nodes:
        target_sections = "".join(
            f"<TARGET ID= '{target_node}'/>" for target_node in edge_targets_by_source[source_node]
        )
        node_sections.append(
            f"<NODE ID= '{source_node}'><TARGETS>{target_sections}</TARGETS></NODE>"
        )
    joined_nodes = "".join(node_sections)
    return f"<ARRAY><NODES>{joined_nodes}</NODES></ARRAY>"


def _format_knowledge_map_as_edge_list(
    edges: list[Edge],
    *,
    use_array_representation: bool,
) -> str:
    if use_array_representation:
        return f"the following list of edges: {edges}"

    edge_tags: list[str] = []
    for source, target in edges:
        edge_tags.append(f"<H>{source}<T>{target}")
    joined_edge_tags = "".join(edge_tags)
    return f"the following RDF representation: <S>{joined_edge_tags}<E>"


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
