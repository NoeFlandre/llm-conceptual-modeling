"""Tests for the shared Mistral client in common/mistral.py.

These tests verify that the MistralChatClient, ChatCompletionClient protocol,
Edge type alias, and knowledge-map formatting helpers are correctly exposed
from the common module.
"""

import json
from types import SimpleNamespace
from typing import cast

import httpx
import pytest

from llm_conceptual_modeling.common.mistral import (
    ChatCompletionClient,
    Edge,
    MistralChatClient,
    _build_adjacency_matrix,
    _build_notation_section,
    _collect_ordered_nodes,
    _format_knowledge_map,
    _format_knowledge_map_as_adjacency,
    _format_knowledge_map_as_edge_list,
    _normalize_structured_response,
    _recover_non_json_response,
)

# ----------------------------------------------------------------------
# Edge type alias
# ----------------------------------------------------------------------


def test_edge_is_a_tuple_of_strings() -> None:
    e: Edge = ("source_node", "target_node")
    assert e[0] == "source_node"
    assert e[1] == "target_node"
    assert isinstance(e, tuple)


# ----------------------------------------------------------------------
# MistralChatClient — integration via fake SDK
# ----------------------------------------------------------------------


def _fake_chat_completion_response(content: str | None) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_mistral_chat_client_complete_json_returns_parsed_dict() -> None:
    """MistralChatClient.complete_json returns a dict parsed from JSON content."""

    class FakeChat:
        def complete(self, **kwargs: object) -> SimpleNamespace:
            return _fake_chat_completion_response(json.dumps({"result": "ok", "count": 42}))

    class FakeSDKClient:
        def __init__(self) -> None:
            self.chat = FakeChat()

    client = MistralChatClient(
        api_key="test-key",
        model="mistral-small-latest",
        sdk_client=FakeSDKClient(),
    )

    result = client.complete_json(
        prompt="test prompt",
        schema_name="test_schema",
        schema={"type": "object", "properties": {}},
    )

    assert result == {"result": "ok", "count": 42}


def test_mistral_chat_client_uses_parse_when_available() -> None:
    captured: dict[str, object] = {}

    class ParsedEdgeList:
        def model_dump(self) -> dict[str, object]:
            return {"edges": [{"source": "alpha", "target": "beta"}]}

    class FakeChat:
        def parse(self, response_format, **kwargs: object) -> SimpleNamespace:
            captured["response_format"] = response_format
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(parsed=ParsedEdgeList()))]
            )

    class FakeSDKClient:
        def __init__(self) -> None:
            self.chat = FakeChat()

    client = MistralChatClient(
        api_key="test-key",
        model="mistral-medium-2508",
        sdk_client=FakeSDKClient(),
    )

    result = client.complete_json(
        prompt="generate edges",
        schema_name="edge_list",
        schema={"type": "object", "properties": {"edges": {"type": "array"}}},
    )

    assert result == {"edges": [{"source": "alpha", "target": "beta"}]}
    assert captured["model"] == "mistral-medium-2508"
    assert captured["messages"] == [{"role": "user", "content": "generate edges"}]
    assert captured["temperature"] == 0.0


def test_mistral_chat_client_falls_back_to_complete_when_parse_fails() -> None:
    captured: dict[str, object] = {}

    class FakeChat:
        def parse(self, **kwargs: object) -> SimpleNamespace:
            raise json.JSONDecodeError("bad json", "['Y', 'N']", 1)

        def complete(self, **kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return _fake_chat_completion_response("['Y', 'N', 'Y']")

    class FakeSDKClient:
        def __init__(self) -> None:
            self.chat = FakeChat()

    client = MistralChatClient(
        api_key="test-key",
        model="mistral-medium-2508",
        sdk_client=FakeSDKClient(),
    )

    result = client.complete_json(
        prompt="verify",
        schema_name="vote_list",
        schema={"type": "object", "properties": {"votes": {"type": "array"}}},
    )

    assert result == {"votes": ["Y", "N", "Y"]}
    assert captured["model"] == "mistral-medium-2508"
    assert captured["messages"] == [{"role": "user", "content": "verify"}]
    assert captured["temperature"] == 0.0


def test_normalize_structured_response_rejects_null_edge_endpoints() -> None:
    with pytest.raises(ValueError, match="null edge target"):
        _normalize_structured_response(
            {"edges": [{"source": "alpha", "target": None}]},
            schema_name="edge_list",
        )


def test_normalize_structured_response_pairs_flat_edge_string_list() -> None:
    actual = _normalize_structured_response(
        ["Prevalence of walking trails", "Physical well-being", "Appetite", "Stress"],
        schema_name="edge_list",
    )

    assert actual == {
        "edges": [
            {
                "source": "Prevalence of walking trails",
                "target": "Physical well-being",
            },
            {"source": "Appetite", "target": "Stress"},
        ]
    }


def test_normalize_structured_response_pairs_flat_edge_scalar_list() -> None:
    actual = _normalize_structured_response(
        [1, 2, "Appetite", "Stress"],
        schema_name="edge_list",
    )

    assert actual == {
        "edges": [
            {"source": "1", "target": "2"},
            {"source": "Appetite", "target": "Stress"},
        ]
    }


def test_normalize_structured_response_accepts_tuple_edge_payload() -> None:
    actual = _normalize_structured_response(
        (("alpha", "beta"), ("gamma", "delta")),
        schema_name="edge_list",
    )

    assert actual == {
        "edges": [
            {"source": "alpha", "target": "beta"},
            {"source": "gamma", "target": "delta"},
        ]
    }


def test_normalize_structured_response_rejects_odd_flat_edge_string_list() -> None:
    with pytest.raises(ValueError, match="even number of items"):
        _normalize_structured_response(
            ["Prevalence of walking trails", "Physical well-being", "Appetite"],
            schema_name="edge_list",
        )


def test_normalize_structured_response_recovers_odd_flat_edge_list_with_noisy_tail() -> None:
    actual = _normalize_structured_response(
        [
            "Prevalence of walking trails",
            "Physical well-being",
            "Appetite",
            "Stress",
            "resil,0,0,0,0",
        ],
        schema_name="edge_list",
    )

    assert actual == {
        "edges": [
            {
                "source": "Prevalence of walking trails",
                "target": "Physical well-being",
            },
            {"source": "Appetite", "target": "Stress"},
        ]
    }


def test_mistral_chat_client_normalizes_list_edge_response() -> None:
    class FakeChat:
        def complete(self, **kwargs: object) -> SimpleNamespace:
            return _fake_chat_completion_response(
                json.dumps([["alpha", "bridge_node"], {"source": "bridge_node", "target": "delta"}])
            )

    class FakeSDKClient:
        def __init__(self) -> None:
            self.chat = FakeChat()

    client = MistralChatClient(
        api_key="test-key",
        model="mistral-medium-2508",
        sdk_client=FakeSDKClient(),
    )

    result = client.complete_json(
        prompt="test prompt",
        schema_name="edge_list",
        schema={"type": "object", "properties": {"edges": {"type": "array"}}},
    )

    assert result == {
        "edges": [
            {"source": "alpha", "target": "bridge_node"},
            {"source": "bridge_node", "target": "delta"},
        ]
    }


def test_mistral_chat_client_recovers_vote_list_from_single_quote_literal() -> None:
    class FakeChat:
        def complete(self, **kwargs: object) -> SimpleNamespace:
            return _fake_chat_completion_response("['Y', 'N', 'Y']")

    class FakeSDKClient:
        def __init__(self) -> None:
            self.chat = FakeChat()

    client = MistralChatClient(
        api_key="test-key",
        model="mistral-medium-2508",
        sdk_client=FakeSDKClient(),
    )

    result = client.complete_json(
        prompt="verify",
        schema_name="vote_list",
        schema={"type": "object", "properties": {"votes": {"type": "array"}}},
    )

    assert result == {"votes": ["Y", "N", "Y"]}


def test_mistral_chat_client_recovers_vote_list_from_bare_tokens() -> None:
    class FakeChat:
        def complete(self, **kwargs: object) -> SimpleNamespace:
            return _fake_chat_completion_response("Y, N, Y")

    class FakeSDKClient:
        def __init__(self) -> None:
            self.chat = FakeChat()

    client = MistralChatClient(
        api_key="test-key",
        model="mistral-medium-2508",
        sdk_client=FakeSDKClient(),
    )

    result = client.complete_json(
        prompt="verify",
        schema_name="vote_list",
        schema={"type": "object", "properties": {"votes": {"type": "array"}}},
    )

    assert result == {"votes": ["Y", "N", "Y"]}


def test_recover_non_json_response_parses_tuple_text_into_edge_dicts() -> None:
    actual = _recover_non_json_response(
        content="(alpha, beta)\n(gamma, delta)",
        schema_name="edge_list",
    )

    assert actual == [
        {"source": "alpha", "target": "beta"},
        {"source": "gamma", "target": "delta"},
    ]


def test_mistral_chat_client_passes_correct_payload_to_sdk() -> None:
    """MistralChatClient sends model, messages, temperature, and response_format."""
    captured: dict[str, object] = {}

    class FakeChat:
        def complete(self, **kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return _fake_chat_completion_response('{"ok":true}')

    class FakeSDKClient:
        def __init__(self) -> None:
            self.chat = FakeChat()

    client = MistralChatClient(
        api_key="test-key",
        model="mistral-medium-2407",
        sdk_client=FakeSDKClient(),
    )

    client.complete_json(
        prompt="count the edges",
        schema_name="edge_count",
        schema={
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        },
    )

    assert captured["model"] == "mistral-medium-2407"
    assert captured["messages"] == [{"role": "user", "content": "count the edges"}]
    assert captured["temperature"] == 0.0
    response_format = cast(dict[str, object], captured["response_format"])
    assert isinstance(response_format, dict)
    assert response_format["type"] == "json_schema"
    json_schema = cast(dict[str, object], response_format["json_schema"])
    assert json_schema["name"] == "edge_count"


def test_mistral_chat_client_retries_on_transient_httpx_connect_error() -> None:
    """MistralChatClient retries on httpx.ConnectError and succeeds on eventual success."""
    attempts = {"n": 0}

    class FakeChat:
        def complete(self, **kwargs: object) -> SimpleNamespace:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise httpx.ConnectError("network unreachable")
            return _fake_chat_completion_response('{"labels":["a","b"]}')

    class FakeSDKClient:
        def __init__(self) -> None:
            self.chat = FakeChat()

    client = MistralChatClient(
        api_key="test-key",
        model="mistral-small-2603",
        sdk_client=FakeSDKClient(),
    )

    result = client.complete_json(
        prompt="expand",
        schema_name="label_list",
        schema={"type": "object", "properties": {"labels": {"type": "array"}}},
    )

    assert result == {"labels": ["a", "b"]}
    assert attempts["n"] == 3


def test_mistral_chat_client_raises_when_content_is_none() -> None:
    """MistralChatClient raises ValueError when the model returns no content."""

    class FakeChat:
        def complete(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))])

    class FakeSDKClient:
        def __init__(self) -> None:
            self.chat = FakeChat()

    client = MistralChatClient(
        api_key="test-key",
        model="mistral-small-2603",
        sdk_client=FakeSDKClient(),
    )

    with pytest.raises(ValueError, match="empty chat completion"):
        client.complete_json(
            prompt="test",
            schema_name="test",
            schema={"type": "object"},
        )


# ----------------------------------------------------------------------
# ChatCompletionClient Protocol
# ----------------------------------------------------------------------


def test_chat_completion_client_protocol_exists() -> None:
    """ChatCompletionClient is a valid Protocol with the expected signature."""

    # duck-typing check: any class implementing complete_json satisfies the protocol
    class MyClient:
        def complete_json(
            self,
            *,
            prompt: str,
            schema_name: str,
            schema: dict[str, object],
        ) -> dict[str, object]:
            return {}

    _: ChatCompletionClient = MyClient()


# ----------------------------------------------------------------------
# Knowledge-map formatting helpers
# ----------------------------------------------------------------------


class TestBuildNotationSection:
    def test_returns_edge_list_text_by_default(self) -> None:
        result = _build_notation_section(
            use_adjacency_notation=False,
            use_array_representation=False,
        )
        assert "directed edges" in result
        assert "adjacency" not in result

    def test_returns_adjacency_matrix_with_tags_when_notation_flag_set(self) -> None:
        result = _build_notation_section(
            use_adjacency_notation=True,
            use_array_representation=False,
        )
        assert "tag-array representation" in result
        assert "<ARRAY><NODES>" in result

    def test_returns_adjacency_matrix_with_arrays_when_both_flags_set(self) -> None:
        result = _build_notation_section(
            use_adjacency_notation=True,
            use_array_representation=True,
        )
        assert "adjacency matrix" in result
        assert "list of nodes" in result

    def test_returns_edge_array_when_only_array_flag_set(self) -> None:
        result = _build_notation_section(
            use_adjacency_notation=False,
            use_array_representation=True,
        )
        assert "arrays of directed edges" in result


class TestCollectOrderedNodes:
    def test_returns_empty_list_for_empty_edges(self) -> None:
        assert _collect_ordered_nodes([]) == []

    def test_returns_source_then_target_in_order_first_seen(self) -> None:
        edges: list[Edge] = [("b", "a"), ("c", "b")]
        assert _collect_ordered_nodes(edges) == ["b", "a", "c"]

    def test_deduplicates_nodes_seen_multiple_times(self) -> None:
        edges: list[Edge] = [("a", "b"), ("b", "c"), ("c", "a")]
        assert _collect_ordered_nodes(edges) == ["a", "b", "c"]


class TestBuildAdjacencyMatrix:
    def test_returns_square_matrix_matching_node_count(self) -> None:
        nodes = ["a", "b", "c"]
        matrix = _build_adjacency_matrix([], ordered_nodes=nodes)
        assert len(matrix) == 3
        assert all(len(row) == 3 for row in matrix)

    def test_marks_correct_positions_for_given_edges(self) -> None:
        nodes = ["a", "b", "c"]
        edges: list[Edge] = [("a", "b"), ("b", "c")]
        matrix = _build_adjacency_matrix(edges, ordered_nodes=nodes)
        assert matrix[0][1] == 1  # a -> b
        assert matrix[1][2] == 1  # b -> c
        assert matrix[0][2] == 0  # a !-> c (no direct edge)


class TestFormatKnowledgeMapAsEdgeList:
    def test_returns_xml_tags_by_default(self) -> None:
        edges: list[Edge] = [("alpha", "beta")]
        result = _format_knowledge_map_as_edge_list(
            edges,
            use_array_representation=False,
        )
        assert result == "the following RDF representation: <S><H>alpha<T>beta<E>"

    def test_returns_python_list_when_use_array_representation(self) -> None:
        edges: list[Edge] = [("alpha", "beta")]
        result = _format_knowledge_map_as_edge_list(
            edges,
            use_array_representation=True,
        )
        assert "alpha" in result
        assert "beta" in result


class TestFormatKnowledgeMapAsAdjacency:
    def test_returns_compact_tag_array_by_default(self) -> None:
        edges: list[Edge] = [("alpha", "beta"), ("beta", "gamma")]
        result = _format_knowledge_map_as_adjacency(
            edges,
            use_array_representation=False,
        )
        assert result.startswith("<ARRAY><NODES><NODE ID= 'alpha'><TARGETS>")
        assert "<TARGET ID= 'beta'/>" in result
        assert "<TARGET ID= 'gamma'/>" in result
        assert "isConnected=" not in result
        assert result.count("<TARGET ID=") == 2

    def test_returns_nodes_and_matrix_when_use_array_representation(self) -> None:
        edges: list[Edge] = [("a", "b")]
        result = _format_knowledge_map_as_adjacency(
            edges,
            use_array_representation=True,
        )
        assert (
            result
            == "the list of nodes ['a', 'b'] and the associated adjacency matrix [[0, 1], [0, 0]]"
        )


class TestFormatKnowledgeMap:
    def test_routes_to_adjacency_when_use_adjacency_notation(self) -> None:
        edges: list[Edge] = [("x", "y")]

        # We use a minimal config-like object to avoid importing the dataclass
        class AdjConfig:
            use_adjacency_notation = True
            use_array_representation = False

        result = _format_knowledge_map(edges, prompt_config=AdjConfig())
        assert result.startswith("<ARRAY><NODES>")

    def test_routes_to_edge_list_when_not_using_adjacency_notation(self) -> None:
        edges: list[Edge] = [("x", "y")]

        class EdgeListConfig:
            use_adjacency_notation = False
            use_array_representation = False

        result = _format_knowledge_map(edges, prompt_config=EdgeListConfig())
        assert result == "the following RDF representation: <S><H>x<T>y<E>"
