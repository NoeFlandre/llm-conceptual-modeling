"""algo1 mistral client — Method 1 (direct edge linking + CoVe verification).

The shared primitives (``MistralChatClient``, ``ChatCompletionClient``,
``Edge``, and the knowledge-map formatting helpers) are imported from
:mod:`llm_conceptual_modeling.common.mistral`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from llm_conceptual_modeling.algo1.cove import apply_cove_verification, build_cove_prompt
from llm_conceptual_modeling.common.mistral import (
    ChatCompletionClient,
    Edge,
    MistralChatClient,
    _build_notation_section,
    _format_knowledge_map,
)

if TYPE_CHECKING:
    pass

__all__ = [
    "ChatCompletionClient",
    "CoveVerifier",
    "Edge",
    "EdgeGenerator",
    "Method1PromptConfig",
    "MistralChatClient",
    "build_cove_verifier",
    "build_direct_edge_prompt",
    "build_edge_generator",
    "extract_vote_list_from_chat_content",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Method1PromptConfig:
    """Prompt-config flags for Method 1 (direct edge linking between two maps)."""

    use_adjacency_notation: bool
    use_array_representation: bool
    include_explanation: bool
    include_example: bool
    include_counterexample: bool


# ---------------------------------------------------------------------------
# algo1-specific example/counterexample sections (edge-linking text)
# ---------------------------------------------------------------------------


def _build_example_section() -> str:
    """algo1-specific example: edge linking between two knowledge maps."""
    return (
        "Here is an example of a desired output for your task. "
        "In knowledge map 1, we have the list of nodes ['capacity to hire', "
        "'bad employees', 'good reputation'] and the associated adjacency matrix "
        "[[0,1,0],[0,0,1],[1,0,0]]. In knowledge map 2, we have the list of nodes "
        "['work motivation', 'productivity', 'financial growth'] and the associated "
        "adjacency matrix [[0,1,0],[0,0,1],[0,0,0]]. In this example, you could "
        "recommend 3 new links: 'quality of managers' with 'work motivation', "
        "'productivity' with 'good reputation' and 'bad employees' with 'quality of managers'. "
        "These links implicitly create 1 new node: 'quality of managers'. Therefore, this is "
        "the expected output: [('quality of managers', 'work motivation'), ('productivity', "
        "'good reputation'), ('bad employees', 'quality of managers')]."
    )


def _build_counterexample_section() -> str:
    """algo1-specific counterexample: edge linking between two knowledge maps."""
    return (
        "Here is an example of a bad output that we do not want to see. "
        "A bad output would be: [('moon', 'bad employees')]. "
        "The error is the recommended link between 'moon' and 'bad employees'. Adding the node "
        "'moon' would be incorrect since it has no relationship with the other nodes. The proposed "
        "link does not represent a true causal relationship."
    )


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_direct_edge_prompt(
    *,
    subgraph1: list[Edge],
    subgraph2: list[Edge],
    prompt_config: Method1PromptConfig | None = None,
) -> str:
    resolved = prompt_config or Method1PromptConfig(
        use_adjacency_notation=False,
        use_array_representation=False,
        include_explanation=False,
        include_example=False,
        include_counterexample=False,
    )
    sections: list[str] = []
    sections.append("You are a helpful assistant who understands Knowledge Maps.")

    if resolved.include_explanation:
        sections.append(
            _build_representation_explanation(
                use_adjacency_notation=resolved.use_adjacency_notation,
                use_array_representation=resolved.use_array_representation,
            )
        )

    sections.append(
        _build_notation_section(
            use_adjacency_notation=resolved.use_adjacency_notation,
            use_array_representation=resolved.use_array_representation,
        )
    )

    if resolved.include_example:
        sections.append(_build_example_section())
    if resolved.include_counterexample:
        sections.append(_build_counterexample_section())

    fmt1 = _format_knowledge_map(subgraph1, prompt_config=resolved)
    fmt2 = _format_knowledge_map(subgraph2, prompt_config=resolved)
    sections.append(f"You will get two inputs: Knowledge map 1: {fmt1} Knowledge map 2: {fmt2}")
    sections.append(
        "Your task is to recommend more links between the two maps. "
        "These links can use new nodes. "
        "Do not suggest links that are already in the maps. "
        "Do not suggest links between nodes of the same map. "
        "Return the recommended links as a list of edges in the format [(A, Z), ..., (X, D)]."
    )
    sections.append(
        "Your output must only be the list of proposed edges. "
        "Do not repeat any instructions I have given you and do not add "
        "unnecessary words or phrases."
    )
    return " ".join(sections)


def _build_representation_explanation(
    *,
    use_adjacency_notation: bool,
    use_array_representation: bool,
) -> str:
    intro = (
        "A knowledge map is a network consisting of nodes and edges. Nodes must have a clear "
        "meaning, such that we can interpret having 'more' or 'less' of a node. "
        "Edges represent the existence of a direct relation between two nodes. "
    )
    if use_adjacency_notation and use_array_representation:
        return (
            intro
            + (
                "The knowledge map is encoded using a list of nodes and an associated adjacency "
                "matrix. "
            )
            + (
                "The adjacency matrix is an n*n square matrix that represents whether each edge "
                "exists. "
            )
            + "In the matrix, each row and each column corresponds to a node. "
            + "Rows and columns come in the same order as the list of nodes. "
            + (
                "A relation between node A and node B is represented as a 1 in the row "
                "corresponding to A "
            )
            + "and the column corresponding to B."
        )

    if use_adjacency_notation and not use_array_representation:
        return intro + (
            "The knowledge map is encoded using tags for nodes and an associated adjacency matrix."
        )

    if not use_adjacency_notation and use_array_representation:
        return (
            intro + "The knowledge map is encoded as a list of edges. Each edge is a pair of nodes."
        )

    return (
        intro + "The knowledge map is encoded using a hierarchical markup language representation."
    )


def extract_vote_list_from_chat_content(content: str) -> list[str]:
    parsed_content = json.loads(content)
    votes = parsed_content["votes"]
    return [str(v) for v in votes]


# ---------------------------------------------------------------------------
# Edge generator & CoVe verifier
# ---------------------------------------------------------------------------


class EdgeGenerator(Protocol):
    def __call__(self, *, subgraph1: list[Edge], subgraph2: list[Edge]) -> list[Edge]: ...


class CoveVerifier(Protocol):
    def __call__(self, candidate_edges: list[Edge]) -> list[Edge]: ...


def build_edge_generator(
    chat_client: ChatCompletionClient,
    prompt_config: "Method1PromptConfig | None" = None,
) -> EdgeGenerator:
    def generate_edges(*, subgraph1: list[Edge], subgraph2: list[Edge]) -> list[Edge]:
        prompt = build_direct_edge_prompt(
            subgraph1=subgraph1, subgraph2=subgraph2, prompt_config=prompt_config
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
        raw_edges: list[dict[str, object]] = cast(list[dict[str, object]], response["edges"])
        return [(str(e["source"]), str(e["target"])) for e in raw_edges]

    return generate_edges


def build_cove_verifier(chat_client: ChatCompletionClient) -> CoveVerifier:
    def verify_edges(candidate_edges: list[Edge]) -> list[Edge]:
        prompt = build_cove_prompt(candidate_edges)
        schema = {
            "type": "object",
            "properties": {"votes": {"type": "array", "items": {"type": "string"}}},
            "required": ["votes"],
            "additionalProperties": False,
        }
        response = chat_client.complete_json(
            prompt=prompt,
            schema_name="vote_list",
            schema=schema,  # type: ignore[arg-type]
        )
        votes: list[str] = cast(list[str], response["votes"])
        return apply_cove_verification(candidate_edges, votes)

    return verify_edges
