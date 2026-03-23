import json
from types import SimpleNamespace

import httpx

from llm_conceptual_modeling.algo2.mistral import (
    Method2PromptConfig,
    MistralChatClient,
    _build_prompt_prefix,
    build_edge_suggester,
    build_edge_suggestion_prompt,
    build_label_expansion_prompt,
    build_label_proposer,
    extract_label_list_from_chat_content,
)


def _fake_chat_completion_response(content: str | None) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_build_label_expansion_prompt_mentions_related_concepts_and_output_format() -> None:
    actual = build_label_expansion_prompt(["alpha", "beta"])

    assert "recommend 5 new related concept names" in actual
    assert "Do not suggest names that are already in the input" in actual
    assert "Return a JSON object" in actual
    assert "alpha" in actual
    assert "beta" in actual


def test_build_label_expansion_prompt_can_include_all_paper_factors() -> None:
    prompt_config = Method2PromptConfig(
        use_adjacency_notation=True,
        use_array_representation=True,
        include_explanation=True,
        include_example=True,
        include_counterexample=True,
        use_relaxed_convergence=True,
    )

    actual = build_label_expansion_prompt(
        ["alpha", "beta"],
        subgraph1=[("alpha", "beta"), ("beta", "gamma")],
        subgraph2=[("delta", "epsilon")],
        prompt_config=prompt_config,
    )

    assert "You are a helpful assistant who can creatively suggest relevant ideas." in actual
    assert "A knowledge map is a network consisting of nodes and edges." in actual
    assert "adjacency matrix" in actual
    assert "Here is an example of a desired output for your task." in actual
    assert "Here is an example of a bad output that we do not want to see." in actual
    assert "Knowledge map 1: {'nodes': ['alpha', 'beta', 'gamma']" in actual
    assert "[[0, 1, 0], [0, 0, 1], [0, 0, 0]]" in actual


def test_build_label_expansion_prompt_can_use_tag_edge_list_without_optional_sections() -> None:
    prompt_config = Method2PromptConfig(
        use_adjacency_notation=False,
        use_array_representation=False,
        include_explanation=False,
        include_example=False,
        include_counterexample=False,
        use_relaxed_convergence=False,
    )

    actual = build_label_expansion_prompt(
        ["alpha", "beta"],
        subgraph1=[("alpha", "beta")],
        subgraph2=[("gamma", "delta")],
        prompt_config=prompt_config,
    )

    assert "A knowledge map is a network consisting of nodes and edges." not in actual
    assert "Here is an example of a desired output for your task." not in actual
    assert "Here is an example of a bad output that we do not want to see." not in actual
    assert "adjacency matrix" not in actual
    expected_map_text = (
        "Knowledge map 1: <knowledge-map><edge source='alpha' target='beta' /></knowledge-map>"
    )
    assert expected_map_text in actual


def test_build_edge_suggestion_prompt_mentions_exact_labels_and_edge_format() -> None:
    actual = build_edge_suggestion_prompt(["alpha", "beta", "bridge_a"])

    assert "suggest edges that directly link concepts" in actual
    assert "Use only exact concept names" in actual
    assert "Return a JSON object" in actual
    assert "bridge_a" in actual


def test_build_edge_suggestion_prompt_can_include_all_paper_factors() -> None:
    prompt_config = Method2PromptConfig(
        use_adjacency_notation=True,
        use_array_representation=True,
        include_explanation=True,
        include_example=True,
        include_counterexample=True,
        use_relaxed_convergence=False,
    )

    actual = build_edge_suggestion_prompt(
        ["alpha", "beta", "bridge_a"],
        subgraph1=[("alpha", "beta"), ("beta", "gamma")],
        subgraph2=[("delta", "epsilon")],
        prompt_config=prompt_config,
    )

    assert "A knowledge map is a network consisting of nodes and edges." in actual
    assert "adjacency matrix" in actual
    assert "Here is an example of a desired output for your task." in actual
    assert "Here is an example of a bad output that we do not want to see." in actual
    assert "Knowledge map 1: {'nodes': ['alpha', 'beta', 'gamma']" in actual
    assert "suggest edges that directly link concepts" in actual


def test_build_edge_suggestion_prompt_can_use_tag_edge_list_without_optional_sections() -> None:
    prompt_config = Method2PromptConfig(
        use_adjacency_notation=False,
        use_array_representation=False,
        include_explanation=False,
        include_example=False,
        include_counterexample=False,
        use_relaxed_convergence=False,
    )

    actual = build_edge_suggestion_prompt(
        ["alpha", "beta", "bridge_a"],
        subgraph1=[("alpha", "beta")],
        subgraph2=[("gamma", "delta")],
        prompt_config=prompt_config,
    )

    assert "A knowledge map is a network consisting of nodes and edges." not in actual
    assert "Here is an example of a desired output for your task." not in actual
    assert "Here is an example of a bad output that we do not want to see." not in actual
    expected_map_text = (
        "Knowledge map 1: <knowledge-map><edge source='alpha' target='beta' /></knowledge-map>"
    )
    assert expected_map_text in actual


def test_build_prompt_prefix_reuses_shared_sections_for_both_prompt_builders() -> None:
    prompt_config = Method2PromptConfig(
        use_adjacency_notation=True,
        use_array_representation=False,
        include_explanation=True,
        include_example=True,
        include_counterexample=True,
        use_relaxed_convergence=True,
    )

    actual = _build_prompt_prefix(prompt_config)

    assert actual == [
        "You are a helpful assistant who can creatively suggest relevant ideas.",
        "A knowledge map is a network consisting of nodes and edges. "
        "Nodes must have a clear meaning, such that we can interpret having "
        "'more' or 'less' of a node. Edges represent the existence of a direct "
        "relation between two nodes.",
        "The knowledge map is encoded using tags for nodes and an associated adjacency matrix.",
        "Here is an example of a desired output for your task. "
        "We have the list of concepts ['capacity to hire', 'bad employees', "
        "'good reputation']. In this example, you could recommend these 9 new concepts: "
        "'employment potential', 'hiring capability', 'staffing ability', "
        "'underperformers', 'inefficient staff', 'problematic workers', "
        "'positive image', 'favorable standing', 'high regard'.",
        "Here is an example of a bad output that we do not want to see. "
        "A bad output would propose unrelated concepts such as 'moon', 'dog', "
        "or 'thermodynamics'.",
    ]


def test_extract_label_list_from_chat_content_supports_json_schema_payload() -> None:
    content = json.dumps({"labels": ["bridge_a", "bridge_b"]})

    actual = extract_label_list_from_chat_content(content)

    assert actual == ["bridge_a", "bridge_b"]


def test_mistral_chat_client_calls_sdk_complete_with_expected_payload() -> None:
    captured_request: dict[str, object] = {}

    class FakeChat:
        def complete(self, **kwargs: object) -> SimpleNamespace:
            captured_request.update(kwargs)
            return _fake_chat_completion_response(json.dumps({"labels": ["bridge_a", "bridge_b"]}))

    class FakeSDKClient:
        def __init__(self) -> None:
            self.chat = FakeChat()

    client = MistralChatClient(
        api_key="test-key",
        model="mistral-small-2603",
        sdk_client=FakeSDKClient(),
    )

    actual = client.complete_json(
        prompt="expand labels",
        schema_name="label_list",
        schema={
            "type": "object",
            "properties": {
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["labels"],
            "additionalProperties": False,
        },
    )

    assert actual == {"labels": ["bridge_a", "bridge_b"]}
    assert captured_request == {
        "model": "mistral-small-2603",
        "messages": [{"role": "user", "content": "expand labels"}],
        "temperature": 0.0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "label_list",
                "schema": {
                    "type": "object",
                    "properties": {
                        "labels": {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["labels"],
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
            return _fake_chat_completion_response(json.dumps({"labels": ["bridge_a", "bridge_b"]}))

    class FakeSDKClient:
        def __init__(self) -> None:
            self.chat = FakeChat()

    client = MistralChatClient(
        api_key="test-key",
        model="mistral-small-2603",
        sdk_client=FakeSDKClient(),
    )

    actual = client.complete_json(
        prompt="expand labels",
        schema_name="label_list",
        schema={
            "type": "object",
            "properties": {
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["labels"],
            "additionalProperties": False,
        },
    )

    assert actual == {"labels": ["bridge_a", "bridge_b"]}
    assert calls["count"] == 3


class FakeChatClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, str | object]] = []

    def complete_json(
        self,
        *,
        prompt: str,
        schema_name: str,
        schema: dict[str, object],
    ) -> dict[str, object]:
        call_record: dict[str, str | object] = {
            "prompt": prompt,
            "schema_name": schema_name,
            "schema": schema,
        }

        self.calls.append(call_record)
        if schema_name == "label_list":
            return {"labels": ["bridge_a", "bridge_b"]}
        return {
            "edges": [
                {"source": "alpha", "target": "bridge_a"},
                {"source": "bridge_b", "target": "beta"},
            ]
        }


def test_build_label_proposer_calls_chat_client_with_label_schema() -> None:
    chat_client = FakeChatClient()

    proposer = build_label_proposer(chat_client)
    actual = proposer(["alpha", "beta"])

    assert actual == ["bridge_a", "bridge_b"]
    assert chat_client.calls[0]["schema_name"] == "label_list"
    assert "recommend 5 new related concept names" in str(chat_client.calls[0]["prompt"])


def test_build_edge_suggester_calls_chat_client_with_edge_schema() -> None:
    chat_client = FakeChatClient()

    suggester = build_edge_suggester(chat_client)
    actual = suggester(["alpha", "beta", "bridge_a", "bridge_b"])

    assert actual == [
        ("alpha", "bridge_a"),
        ("bridge_b", "beta"),
    ]
    assert chat_client.calls[0]["schema_name"] == "edge_list"
    assert "suggest edges that directly link concepts" in str(chat_client.calls[0]["prompt"])
