"""algo2 mistral client — Method 2 (label expansion + edge suggestion).

The shared primitives (``MistralChatClient``, ``ChatCompletionClient``,
``Edge``, and the knowledge-map formatting helpers) are imported from
:mod:`llm_conceptual_modeling.common.mistral`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from llm_conceptual_modeling.common.mistral import (
    ChatCompletionClient,
    Edge,
    MistralChatClient,
    _format_knowledge_map,
)

if TYPE_CHECKING:
    pass

__all__ = [
    "ChatCompletionClient",
    "Edge",
    "EdgeSuggester",
    "LabelProposer",
    "Method2PromptConfig",
    "MistralChatClient",
    "_build_prompt_prefix",
    "build_edge_suggester",
    "build_edge_suggestion_prompt",
    "build_label_expansion_prompt",
    "build_label_proposer",
    "extract_label_list_from_chat_content",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Method2PromptConfig:
    """Prompt-config flags for Method 2 (label expansion + edge suggestion)."""

    use_adjacency_notation: bool
    use_array_representation: bool
    include_explanation: bool
    include_example: bool
    include_counterexample: bool
    use_relaxed_convergence: bool


def _resolve_prompt_config(prompt_config: Method2PromptConfig | None) -> Method2PromptConfig:
    return prompt_config or Method2PromptConfig(
        use_adjacency_notation=False,
        use_array_representation=False,
        include_explanation=False,
        include_example=False,
        include_counterexample=False,
        use_relaxed_convergence=False,
    )


def _build_prompt_prefix(prompt_config: Method2PromptConfig) -> list[str]:
    """Return the shared opening sections used by both Method 2 prompts."""
    sections: list[str] = ["You are a helpful assistant who understands Knowledge Maps."]

    if prompt_config.include_explanation:
        sections.append(
            "A knowledge map is a network consisting of nodes and edges. "
            "Nodes must have a clear meaning, such that we can interpret having "
            "'more' or 'less' of a node. Edges represent the existence of a direct "
            "relation between two nodes."
        )

    sections.append(_build_notation_section(prompt_config))

    if prompt_config.include_example:
        sections.append(_build_example_section(prompt_config))
    if prompt_config.include_counterexample:
        sections.append(_build_counterexample_section(prompt_config))

    return sections


def _build_notation_section(prompt_config: Method2PromptConfig) -> str:
    if prompt_config.use_adjacency_notation and prompt_config.use_array_representation:
        return (
            "The knowledge map is encoded using a list of nodes and an associated adjacency "
            "matrix. The adjacency matrix is an n*n square matrix that represents whether each "
            "edge exists. In the matrix, each row and each column corresponds to a node. Rows "
            "and columns come in the same order as the list of nodes. A relation between node A "
            "and node B is represented as a 1 in the row corresponding to A and the column "
            "corresponding to B."
        )

    if prompt_config.use_adjacency_notation:
        return (
            "The knowledge map is encoded using a tag-array representation. The map is wrapped "
            "in <ARRAY><NODES>...</NODES></ARRAY>. Each <NODE> stores its ID and only the "
            "outgoing targets that are directly connected to it."
        )

    if prompt_config.use_array_representation:
        return (
            "The knowledge map is encoded using the RDF representation. The RDF representation "
            "uses tags denoted by <>. <S> is the start of the map. <E> is the end of the map. "
            "In between <S> and <E>, we list all edges of the map. Each edge is represented with "
            "two tags: <H> precedes the node, then <T> precedes the target node."
        )

    return "The knowledge map is encoded as a list of edges. Each edge is a pair of nodes."


def _build_example_section(prompt_config: Method2PromptConfig) -> str:
    if prompt_config.use_adjacency_notation and prompt_config.use_array_representation:
        return (
            "Here is an example of a desired output for your task. In knowledge map 1, we have "
            "the list of nodes ['capacity to hire', 'bad employees', 'good reputation'] and the "
            "associated adjacency matrix [[0,1,0],[0,0,1],[1,0,0]]. In knowledge map 2, we have "
            "the list of nodes ['work motivation', 'productivity', 'financial growth'] and the "
            "associated adjacency matrix [[0,1,0],[0,0,1],[0,0,0]]. In this example, you could "
            "recommend these 5 new nodes: 'quality of managers', 'employee satisfaction', "
            "'customer satisfaction', 'market share', 'performance incentives'. Therefore, this "
            "is the expected output: ['quality of managers', 'employee satisfaction', "
            "'customer satisfaction', 'market share', 'performance incentives']."
        )

    if prompt_config.use_adjacency_notation:
        return (
            "Here is an example of a desired output for your task. In knowledge map 1, we have "
            "the following tag-array representation: <ARRAY><NODES><NODE ID= 'capacity to hire'>"
            "<TARGETS><TARGET ID= 'bad employees'/></TARGETS></NODE><NODE ID= 'bad employees'>"
            "<TARGETS><TARGET ID= 'good reputation'/></TARGETS></NODE><NODE ID= 'good reputation'>"
            "<TARGETS><TARGET ID= 'capacity to hire'/></TARGETS></NODE></NODES></ARRAY>. In "
            "knowledge map 2, we have the following tag-array representation: <ARRAY><NODES>"
            "<NODE ID= 'work motivation'><TARGETS><TARGET ID= 'productivity'/></TARGETS></NODE>"
            "<NODE ID= 'productivity'><TARGETS><TARGET ID= 'financial growth'/></TARGETS></NODE>"
            "<NODE ID= 'financial growth'><TARGETS></TARGETS></NODE></NODES></ARRAY>. In this "
            "example, you could recommend these 5 new nodes: 'quality of managers', "
            "'employee satisfaction', 'customer satisfaction', 'market share', "
            "'performance incentives'. Therefore, this is the expected output: "
            "['quality of managers', 'employee satisfaction', 'customer satisfaction', "
            "'market share', 'performance incentives']."
        )

    if prompt_config.use_array_representation:
        return (
            "Here is an example of a desired output for your task. In knowledge map 1, we have "
            "the following RDF representation: <S><H>capacity to hire<T>bad employees<H>bad "
            "employees<T>good reputation<H>good reputation<T>capacity to hire<E>. In knowledge "
            "map 2, we have the following RDF representation: <S><H>work motivation<T>productivity"
            "<H>productivity<T>financial growth<E>. In this example, you could recommend these 5 "
            "new nodes: 'quality of managers', 'employee satisfaction', 'customer satisfaction', "
            "'market share', 'performance incentives'. Therefore, this is the expected output: "
            "['quality of managers', 'employee satisfaction', 'customer satisfaction', 'market "
            "share', 'performance incentives']."
        )

    return (
        "Here is an example of a desired output for your task. In knowledge map 1, we have the "
        "following list of edges: [('capacity to hire', 'bad employees'), ('bad employees', "
        "'good reputation'), ('good reputation', 'capacity to hire')]. In knowledge map 2, we "
        "have the following list of edges: [('work motivation', 'productivity'), "
        "('productivity', 'financial growth')]. In this example, you could recommend these 5 "
        "new nodes: 'quality of managers', 'employee satisfaction', 'customer satisfaction', "
        "'market share', 'performance incentives'. Therefore, this is the expected output: "
        "['quality of managers', 'employee satisfaction', 'customer satisfaction', 'market "
        "share', 'performance incentives']."
    )


def _build_counterexample_section(prompt_config: Method2PromptConfig) -> str:
    if prompt_config.use_adjacency_notation and prompt_config.use_array_representation:
        return (
            "Here is an example of a bad output that we do not want to see. In knowledge map 1, "
            "we have the list of nodes ['capacity to hire', 'bad employees', 'good reputation'] "
            "and the associated adjacency matrix [[0,1,0],[0,0,1],[1,0,0]]. In knowledge map 2, "
            "we have the list of nodes ['work motivation', 'productivity', 'financial growth'] "
            "and the associated adjacency matrix [[0,1,0],[0,0,1],[0,0,0]]. A bad output would "
            "be: ['moon', 'dog', 'thermodynamics', 'swimming', 'red']. Adding the proposed nodes "
            "would be incorrect since they have no relationship with the nodes in the input."
        )

    if prompt_config.use_adjacency_notation:
        return (
            "Here is an example of a bad output that we do not want to see. In knowledge map 1, "
            "we have the following tag-array representation: <ARRAY><NODES><NODE ID= "
            "'capacity to hire'><TARGETS><TARGET ID= 'bad employees'/></TARGETS></NODE><NODE "
            "ID= 'bad employees'><TARGETS><TARGET ID= 'good reputation'/></TARGETS></NODE><NODE "
            "ID= 'good reputation'><TARGETS><TARGET ID= 'capacity to hire'/></TARGETS></NODE>"
            "</NODES></ARRAY>. In knowledge map 2, we have the following tag-array "
            "representation: <ARRAY><NODES><NODE ID= 'work motivation'><TARGETS><TARGET ID= "
            "'productivity'/></TARGETS></NODE><NODE ID= 'productivity'><TARGETS><TARGET ID= "
            "'financial growth'/></TARGETS></NODE><NODE ID= 'financial growth'><TARGETS>"
            "</TARGETS></NODE></NODES></ARRAY>. A bad output would be: ['moon', 'dog', "
            "'thermodynamics', 'swimming', 'red']. Adding the proposed nodes would be "
            "incorrect since they have no relationship with the nodes in the input."
        )

    if prompt_config.use_array_representation:
        return (
            "Here is an example of a bad output that we do not want to see. In knowledge map 1, "
            "we have the following RDF representation: <S><H>capacity to hire<T>bad employees<H>"
            "bad employees<T>good reputation<H>good reputation<T>capacity to hire<E>. In knowledge "
            "map 2, we have the following RDF representation: <S><H>work motivation<T>productivity"
            "<H>productivity<T>financial growth<E>. A bad output would be: ['moon', 'dog', "
            "'thermodynamics', 'swimming', 'red']. Adding the proposed nodes would be incorrect "
            "since they have no relationship with the nodes in the input."
        )

    return (
        "Here is an example of a bad output that we do not want to see. In knowledge map 1, we "
        "have the following list of edges: [('capacity to hire', 'bad employees'), "
        "('bad employees', 'good reputation'), ('good reputation', 'capacity to hire')]. In "
        "knowledge map 2, we have the following list of edges: [('work motivation', "
        "'productivity'), ('productivity', 'financial growth')]. A bad output would be: "
        "['moon', 'dog', 'thermodynamics', 'swimming', 'red']. Adding the proposed nodes would "
        "be incorrect since they have no relationship with the nodes in the input."
    )


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_label_expansion_prompt(
    seed_labels: list[str],
    *,
    subgraph1: list[Edge] | None = None,
    subgraph2: list[Edge] | None = None,
    prompt_config: Method2PromptConfig | None = None,
) -> str:
    resolved = _resolve_prompt_config(prompt_config)
    sections: list[str] = _build_prompt_prefix(resolved)

    if subgraph1 is not None and subgraph2 is not None:
        fmt1 = _format_knowledge_map(subgraph1, prompt_config=resolved)
        fmt2 = _format_knowledge_map(subgraph2, prompt_config=resolved)
        sections.append(f"You will get two inputs: Knowledge map 1: {fmt1} Knowledge map 2: {fmt2}")
    else:
        sections.append(f"Input concepts: {', '.join(seed_labels)}")

    sections.append(
        "Your task is to recommend 5 more nodes in relation to those already in the two "
        "knowledge maps. Do not suggest nodes that are already in the maps. Return the "
        "recommended nodes as a list of nodes in the format ['A', 'B', 'C', 'D', 'E']."
    )
    sections.append(
        "Your output must only be the list of proposed concepts. "
        "Do not repeat any instructions I have given you and do not add "
        "unnecessary words or phrases."
    )
    return " ".join(sections)


def build_edge_suggestion_prompt(
    expanded_label_context: list[str],
    *,
    subgraph1: list[Edge] | None = None,
    subgraph2: list[Edge] | None = None,
    prompt_config: Method2PromptConfig | None = None,
) -> str:
    resolved = _resolve_prompt_config(prompt_config)
    sections: list[str] = _build_prompt_prefix(resolved)

    if subgraph1 is not None and subgraph2 is not None:
        fmt1 = _format_knowledge_map(subgraph1, prompt_config=resolved)
        fmt2 = _format_knowledge_map(subgraph2, prompt_config=resolved)
        sections.append(f"You will get two inputs: Knowledge map 1: {fmt1} Knowledge map 2: {fmt2}")

    sections.append(
        "Your task is to recommend more links between the two maps. These links can use new "
        "nodes. Do not suggest links that are already in the maps. Do not suggest links between "
        "nodes of the same map. Return the recommended links as a list of edges in the format "
        "[(A, Z), ..., (X, D)]."
    )
    sections.append(f"Available concepts: {', '.join(expanded_label_context)}")
    return " ".join(sections)


def extract_label_list_from_chat_content(content: str) -> list[str]:
    parsed_content = json.loads(content)
    labels = parsed_content["labels"]
    return [str(label) for label in labels]


# ---------------------------------------------------------------------------
# Proposers
# ---------------------------------------------------------------------------


class LabelProposer(Protocol):
    def __call__(self, current_labels: list[str]) -> list[str]: ...


class EdgeSuggester(Protocol):
    def __call__(self, expanded_label_context: list[str]) -> list[Edge]: ...


def build_label_proposer(
    chat_client: ChatCompletionClient,
    prompt_config: Method2PromptConfig | None = None,
) -> LabelProposer:
    def propose(current_labels: list[str]) -> list[str]:
        prompt = build_label_expansion_prompt(
            current_labels,
            prompt_config=prompt_config,
        )
        schema = {
            "type": "object",
            "properties": {"labels": {"type": "array", "items": {"type": "string"}}},
            "required": ["labels"],
            "additionalProperties": False,
        }
        response = chat_client.complete_json(
            prompt=prompt,
            schema_name="label_list",
            schema=schema,  # type: ignore[arg-type]
        )
        return [str(label) for label in cast(list[str], response["labels"])]

    return propose


def build_edge_suggester(
    chat_client: ChatCompletionClient,
    prompt_config: Method2PromptConfig | None = None,
) -> EdgeSuggester:
    def suggest(expanded_label_context: list[str]) -> list[Edge]:
        prompt = build_edge_suggestion_prompt(
            expanded_label_context,
            prompt_config=prompt_config,
        )
        schema = {
            "type": "object",
            "properties": {
                "edges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"source": {"type": "string"}, "target": {"type": "string"}},
                        "required": ["source", "target"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["edges"],
            "additionalProperties": False,
        }
        response = chat_client.complete_json(
            prompt=prompt,
            schema_name="edge_list",
            schema=schema,  # type: ignore[arg-type]
        )
        raw_edges = cast(list[dict[str, object]], response["edges"])
        return [(str(e["source"]), str(e["target"])) for e in raw_edges]

    return suggest
