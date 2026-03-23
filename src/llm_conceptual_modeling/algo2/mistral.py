"""algo2 mistral client — Method 2 (label expansion + edge suggestion).

The shared primitives (``MistralChatClient``, ``ChatCompletionClient``,
``Edge``, and the knowledge-map formatting helpers) are imported from
:mod:`llm_conceptual_modeling.common.mistral`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from llm_conceptual_modeling.common.mistral import (
    ChatCompletionClient,
    Edge,
    MistralChatClient,
    _build_counterexample_section,
    _build_example_section,
    _build_notation_section,
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
    sections: list[str] = ["You are a helpful assistant who can creatively suggest relevant ideas."]

    if prompt_config.include_explanation:
        sections.append(
            "A knowledge map is a network consisting of nodes and edges. "
            "Nodes must have a clear meaning, such that we can interpret having "
            "'more' or 'less' of a node. Edges represent the existence of a direct "
            "relation between two nodes."
        )

    sections.append(
        _build_notation_section(
            use_adjacency_notation=prompt_config.use_adjacency_notation,
            use_array_representation=prompt_config.use_array_representation,
        )
    )

    if prompt_config.include_example:
        sections.append(_build_example_section())
    if prompt_config.include_counterexample:
        sections.append(_build_counterexample_section())

    return sections


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
        "Your task is to recommend 5 new related concept names for the input concepts. "
        "Do not suggest names that are already in the input. "
        "Return a JSON object with a top-level 'labels' array containing only strings."
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
        f"Given a set of concept names, suggest edges that directly link concepts "
        f"using the provided list. Use only exact concept names from the provided list. "
        f"Return a JSON object with a top-level 'edges' array. "
        f"Each edge must be an object with 'source' and 'target' string fields. "
        f"Available concepts: {', '.join(expanded_label_context)}"
    )
    return " ".join(sections)


def extract_label_list_from_chat_content(content: str) -> list[str]:
    parsed_content = json.loads(content)
    labels = parsed_content["labels"]
    return [str(label) for label in labels]


# ---------------------------------------------------------------------------
# Proposers
# ---------------------------------------------------------------------------


class LabelProposer:
    def __call__(self, current_labels: list[str]) -> list[str]: ...


class EdgeSuggester:
    def __call__(self, expanded_label_context: list[str]) -> list[Edge]: ...


def build_label_proposer(chat_client: ChatCompletionClient) -> LabelProposer:
    def propose(current_labels: list[str]) -> list[str]:
        prompt = build_label_expansion_prompt(current_labels)
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


def build_edge_suggester(chat_client: ChatCompletionClient) -> EdgeSuggester:
    def suggest(expanded_label_context: list[str]) -> list[Edge]:
        prompt = build_edge_suggestion_prompt(expanded_label_context)
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
