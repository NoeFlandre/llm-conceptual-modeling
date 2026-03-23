"""algo3 mistral client — Method 3 (tree-structured label expansion).

The shared primitives (``MistralChatClient`` and ``ChatCompletionClient``)
are imported from :mod:`llm_conceptual_modeling.common.mistral`.
algo3 does not use knowledge maps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from llm_conceptual_modeling.common.mistral import ChatCompletionClient, MistralChatClient

if TYPE_CHECKING:
    pass

__all__ = [
    "ChatCompletionClient",
    "ChildProposer",
    "Method3PromptConfig",
    "MistralChatClient",
    "build_child_proposer",
    "build_tree_expansion_prompt",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Method3PromptConfig:
    """Prompt-config flags for Method 3 (tree-structured expansion)."""

    include_example: bool
    include_counterexample: bool


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def build_tree_expansion_prompt(
    *,
    source_labels: list[str],
    child_count: int,
    prompt_config: Method3PromptConfig | None = None,
) -> str:
    resolved = prompt_config or Method3PromptConfig(
        include_example=False,
        include_counterexample=False,
    )
    sections: list[str] = []
    sections.append("You are a helpful assistant who can creatively suggest relevant ideas.")
    sections.append(
        "Your input is a set of concept names. All concept names must have a clear meaning, such "
        "that we can interpret having 'more' or 'less' of a concept."
    )
    sections.append("Your input is the following list of concept names:")
    sections.append(str(source_labels))
    sections.append(
        "Your task is to recommend {} related concept names for each of the names in the input. "
        "Do not suggest names that are in the input. Your output must include the list of the {} "
        "proposed names for each of the input names. Do not include any other text. Return your "
        "proposed names in a dictionary format {{ 'A' : ['B' , 'C', 'D'], 'E' : ['F' , 'G' , "
        "'H'], …, 'U' : ['V' , 'W' , 'X'] }}.".format(child_count, child_count)
    )
    if resolved.include_example:
        sections.append(_build_example_section(child_count))
    if resolved.include_counterexample:
        sections.append(_build_counterexample_section(child_count))
    sections.append(
        "Your output must only be the list of proposed concepts. Do not repeat any instructions I "
        "have given you and do not add unnecessary words or phrases."
    )
    return " ".join(sections)


def _build_example_section(child_count: int) -> str:
    if child_count == 3:
        return (
            "Here is an example of a desired output for your task. We have the list of concepts "
            "['capacity to hire', 'bad employees', 'good reputation']. In this example, you could "
            "recommend these 9 new concepts: 'employment potential', 'hiring capability', "
            "'staffing ability', 'underperformers', 'inefficient staff', 'problematic workers', "
            "'positive image', 'favorable standing', 'high regard'. Therefore, this is the "
            "expected output: {{ \"capacity to hire\": ['employment potential', 'hiring capability', "
            "'staffing ability'], \"bad employees\": ['underperformers', 'inefficient staff', "
            "'problematic workers'], \"good reputation\": ['positive image', 'favorable standing', "
            "'high regard'] }}."
        )

    return (
        "Here is an example of a desired output for your task. We have the list of concepts "
        "['capacity to hire', 'bad employees', 'good reputation']. In this example, you could "
        "recommend these 15 new concepts: 'employment potential', 'hiring capability', "
        "'staffing ability', 'underperformers', 'inefficient staff', 'problematic workers', "
        "'positive image', 'favorable standing', 'high regard'. Therefore, this is the expected "
        'output: { "capacity to hire": ["employment potential", "hiring capability", '
        '"staffing ability", "recruitment capacity", "talent acquisition"], '
        '"bad employees": ["underperformers", "inefficient staff", "problematic workers", '
        '"low performers", "unproductive staff"], "good reputation": ["positive image", '
        '"favorable standing", "high regard", "excellent reputation", "commendable status"] }.'
    )


def _build_counterexample_section(child_count: int) -> str:
    if child_count == 3:
        return (
            "Here is an example of a bad output that we do not want to see. We have the list of "
            "nodes ['capacity to hire', 'bad employees', 'good reputation']. A bad output would "
            "be: {{ \"capacity to hire\": ['moon', 'dog', 'thermodynamics'], \"bad employees\": "
            "['swimming', 'red', 'happiness'], \"good reputation\": ['judo', 'canada', 'light'] "
            "}}. Adding the proposed concepts would be incorrect since they have no relationship "
            "with the concepts in the input."
        )

    return (
        "Here is an example of a bad output that we do not want to see. We have the list of "
        "nodes ['capacity to hire', 'bad employees', 'good reputation']. A bad output would be: "
        "{ \"capacity to hire\": ['moon', 'dog', 'thermodynamics', 'country', 'pillow'], "
        "\"bad employees\": ['swimming', 'red', 'happiness', 'food', 'shoe'], "
        "\"good reputation\": ['judo', 'canada', 'light', 'phone', 'electricity'] }. Adding the "
        "proposed concepts would be incorrect since they have no relationship with the concepts "
        "in the input."
    )


# ---------------------------------------------------------------------------
# Child proposer
# ---------------------------------------------------------------------------


class ChildProposer:
    def __call__(
        self,
        source_labels: list[str],
        *,
        child_count: int,
    ) -> dict[str, list[str]]: ...


def build_child_proposer(
    chat_client: ChatCompletionClient,
    prompt_config: Method3PromptConfig | None = None,
) -> ChildProposer:
    def propose_children(
        source_labels: list[str],
        *,
        child_count: int,
    ) -> dict[str, list[str]]:
        prompt = build_tree_expansion_prompt(
            source_labels=source_labels,
            child_count=child_count,
            prompt_config=prompt_config,
        )
        schema = {
            "type": "object",
            "properties": {
                "children_by_label": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                }
            },
            "required": ["children_by_label"],
            "additionalProperties": False,
        }
        response = chat_client.complete_json(
            prompt=prompt,
            schema_name="children_by_label",
            schema=schema,  # type: ignore[arg-type]
        )
        raw = cast(dict[str, list[str]], response["children_by_label"])
        return {str(k): [str(v) for v in vs] for k, vs in raw.items()}

    return propose_children
