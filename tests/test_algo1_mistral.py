import json
from types import SimpleNamespace

import httpx

from llm_conceptual_modeling.algo1.mistral import (
    Method1PromptConfig,
    MistralChatClient,
    build_cove_verifier,
    build_direct_edge_prompt,
    build_edge_generator,
    extract_vote_list_from_chat_content,
)


def _fake_chat_completion_response(content: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_build_direct_edge_prompt_mentions_new_nodes_and_output_format() -> None:
    actual = build_direct_edge_prompt(
        subgraph1=[("alpha", "beta")],
        subgraph2=[("gamma", "delta")],
    )

    assert "recommend more links between the two maps" in actual
    assert "These links can use new nodes" in actual
    assert "Return a JSON object" in actual
    assert "alpha" in actual
    assert "gamma" in actual


def test_build_direct_edge_prompt_can_include_all_paper_factors() -> None:
    prompt_config = Method1PromptConfig(
        use_adjacency_notation=True,
        use_array_representation=True,
        include_explanation=True,
        include_example=True,
        include_counterexample=True,
    )

    actual = build_direct_edge_prompt(
        subgraph1=[("alpha", "beta"), ("beta", "gamma")],
        subgraph2=[("delta", "epsilon")],
        prompt_config=prompt_config,
    )

    assert "You are a helpful assistant who understands Knowledge Maps." in actual
    assert "A knowledge map is a network consisting of nodes and edges." in actual
    assert "adjacency matrix" in actual
    assert "Here is an example of a desired output for your task." in actual
    assert "Here is an example of a bad output that we do not want to see." in actual
    assert "Knowledge map 1: {'nodes': ['alpha', 'beta', 'gamma']" in actual
    assert "[[0, 1, 0], [0, 0, 1], [0, 0, 0]]" in actual


def test_build_direct_edge_prompt_can_use_edge_list_without_optional_sections() -> None:
    prompt_config = Method1PromptConfig(
        use_adjacency_notation=False,
        use_array_representation=False,
        include_explanation=False,
        include_example=False,
        include_counterexample=False,
    )

    actual = build_direct_edge_prompt(
        subgraph1=[("alpha", "beta")],
        subgraph2=[("gamma", "delta")],
        prompt_config=prompt_config,
    )

    assert "A knowledge map is a network consisting of nodes and edges." not in actual
    assert "Here is an example of a desired output for your task." not in actual
    assert "Here is an example of a bad output that we do not want to see." not in actual
    assert "adjacency matrix" not in actual
    expected_map_text = (
        "Knowledge map 1: "
        "<knowledge-map><edge source='alpha' target='beta' /></knowledge-map>"
    )
    assert expected_map_text in actual


def test_extract_vote_list_from_chat_content_supports_json_schema_payload() -> None:
    content = json.dumps({"votes": ["Y", "N", "Y"]})

    actual = extract_vote_list_from_chat_content(content)

    assert actual == ["Y", "N", "Y"]


def test_mistral_chat_client_calls_sdk_complete_with_expected_payload() -> None:
    captured_request: dict[str, object] = {}

    class FakeChat:
        def complete(self, **kwargs: object) -> SimpleNamespace:
            captured_request.update(kwargs)
            return _fake_chat_completion_response(
                json.dumps({"edges": [{"source": "a", "target": "b"}]})
            )

    class FakeSDKClient:
        def __init__(self) -> None:
            self.chat = FakeChat()

    client = MistralChatClient(
        api_key="test-key",
        model="mistral-small-2603",
        sdk_client=FakeSDKClient(),
    )

    actual = client.complete_json(
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
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["edges"],
            "additionalProperties": False,
        },
    )

    assert actual == {"edges": [{"source": "a", "target": "b"}]}
    assert captured_request == {
        "model": "mistral-small-2603",
        "messages": [{"role": "user", "content": "generate edges"}],
        "temperature": 0.0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "edge_list",
                "schema": {
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
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["edges"],
                    "additionalProperties": False,
                },
            },
        },
    }


def test_mistral_chat_client_retries_transient_transport_errors() -> None:
    calls = {"count": 0}

    class FakeChat:
        def complete(self, **kwargs: object) -> SimpleNamespace:
            calls["count"] += 1
            if calls["count"] < 3:
                raise httpx.ConnectError("temporary network issue")
            return _fake_chat_completion_response(
                json.dumps({"edges": [{"source": "a", "target": "b"}]})
            )

    class FakeSDKClient:
        def __init__(self) -> None:
            self.chat = FakeChat()

    client = MistralChatClient(
        api_key="test-key",
        model="mistral-small-2603",
        sdk_client=FakeSDKClient(),
    )

    actual = client.complete_json(
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
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["edges"],
            "additionalProperties": False,
        },
    )

    assert actual == {"edges": [{"source": "a", "target": "b"}]}
    assert calls["count"] == 3


class FakeChatClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def complete_json(
        self,
        *,
        prompt: str,
        schema_name: str,
        schema: dict[str, object],
    ) -> dict[str, object]:
        self.calls.append({"prompt": prompt, "schema_name": schema_name})
        if schema_name == "edge_list":
            return {
                "edges": [
                    {"source": "alpha", "target": "bridge_a"},
                    {"source": "bridge_b", "target": "gamma"},
                ]
            }
        return {"votes": ["Y", "N"]}


def test_build_edge_generator_calls_chat_client_with_edge_schema() -> None:
    chat_client = FakeChatClient()

    generator = build_edge_generator(chat_client)
    actual = generator(
        subgraph1=[("alpha", "beta")],
        subgraph2=[("gamma", "delta")],
    )

    assert actual == [
        ("alpha", "bridge_a"),
        ("bridge_b", "gamma"),
    ]
    assert chat_client.calls[0]["schema_name"] == "edge_list"


def test_build_cove_verifier_filters_edges_using_vote_schema() -> None:
    chat_client = FakeChatClient()

    verifier = build_cove_verifier(chat_client)
    actual = verifier(
        [
            ("alpha", "bridge_a"),
            ("bridge_b", "gamma"),
        ]
    )

    assert actual == [("alpha", "bridge_a")]
    assert chat_client.calls[0]["schema_name"] == "vote_list"
