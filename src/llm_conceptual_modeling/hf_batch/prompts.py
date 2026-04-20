from __future__ import annotations

from typing import Any, cast

from llm_conceptual_modeling.algo1.cove import apply_cove_verification, build_cove_prompt
from llm_conceptual_modeling.hf_batch.types import Edge


def build_prompt_bundle(
    *,
    algorithm_name: str,
    algorithm_config: Any,
    active_high_factors: list[str] | None = None,
    prompt_factors: dict[str, bool | int],
) -> dict[str, str]:
    _ = active_high_factors
    if algorithm_name == "algo1":
        return _build_algo1_prompt_bundle(algorithm_config, prompt_factors=prompt_factors)
    if algorithm_name == "algo2":
        return _build_algo2_prompt_bundle(algorithm_config, prompt_factors=prompt_factors)
    return _build_algo3_prompt_bundle(algorithm_config, prompt_factors=prompt_factors)


def _build_algo1_prompt_bundle(
    algorithm_config: Any,
    *,
    prompt_factors: dict[str, bool | int],
) -> dict[str, str]:
    definitions = algorithm_config.fragment_definitions
    if "system_message" not in definitions:
        direct_edge_template = _select_template_name(
            algorithm_config,
            preferred="direct_edge",
        )
        cove_template = _select_template_name(
            algorithm_config,
            preferred="cove_verification",
        )
        return {
            "direct_edge": algorithm_config.assemble_prompt([], template_name=direct_edge_template),
            "cove_verification": algorithm_config.assemble_prompt([], template_name=cove_template),
        }
    variant = _resolve_representation_variant(prompt_factors)
    explanation_sections: list[str] = []
    if bool(prompt_factors.get("include_explanation", False)):
        explanation_sections = [
            definitions["explanation_fixed"],
            definitions[f"explanation_variable_{variant}"],
        ]
    example_section = (
        definitions[f"example_variable_{variant}"]
        if bool(prompt_factors.get("include_example", False))
        else ""
    )
    counterexample_section = (
        definitions[f"counter_example_variable_{variant}"]
        if bool(prompt_factors.get("include_counterexample", False))
        else ""
    )
    direct_edge_prompt = _join_prompt_sections(
        definitions["system_message"],
        *explanation_sections,
        example_section,
        counterexample_section,
        f"{definitions['task_fixed_sub_1']} {{formatted_subgraph1}}",
        f"{definitions['task_fixed_sub_2']} {{formatted_subgraph2}}",
        definitions["task_fixed_sub_3"],
        definitions["conclusion_fixed"],
    )
    return {
        "direct_edge": direct_edge_prompt,
        "cove_verification": definitions["cove_verification_template"],
    }


def _build_algo2_prompt_bundle(
    algorithm_config: Any,
    *,
    prompt_factors: dict[str, bool | int],
) -> dict[str, str]:
    definitions = algorithm_config.fragment_definitions
    if "system_message" not in definitions:
        label_template = _select_template_name(
            algorithm_config,
            preferred="label_expansion",
        )
        edge_template = _select_template_name(
            algorithm_config,
            preferred="edge_suggestion",
        )
        cove_template = _select_template_name(
            algorithm_config,
            preferred="cove_verification",
        )
        return {
            "label_expansion": algorithm_config.assemble_prompt(
                [],
                template_name=label_template,
            ),
            "edge_suggestion": algorithm_config.assemble_prompt(
                [],
                template_name=edge_template,
            ),
            "cove_verification": algorithm_config.assemble_prompt(
                [],
                template_name=cove_template,
            ),
        }
    variant = _resolve_representation_variant(prompt_factors)
    explanation_sections: list[str] = []
    if bool(prompt_factors.get("include_explanation", False)):
        explanation_sections = [
            definitions["explanation_fixed"],
            definitions[f"explanation_variable_{variant}"],
        ]
    example_section = (
        definitions[f"example_variable_{variant}"]
        if bool(prompt_factors.get("include_example", False))
        else ""
    )
    counterexample_section = (
        definitions[f"counter_example_variable_{variant}"]
        if bool(prompt_factors.get("include_counterexample", False))
        else ""
    )
    label_expansion_prompt = _join_prompt_sections(
        definitions["system_message"],
        *explanation_sections,
        example_section,
        counterexample_section,
        f"{definitions['task_fixed_sub_1']} {{formatted_subgraph1}}",
        f"{definitions['task_fixed_sub_2']} {{formatted_subgraph2}}",
        definitions.get("iterative_context_fragment", ""),
        definitions["task_fixed_sub_3"],
        definitions["conclusion_fixed"],
    )
    edge_suggestion_prompt = _join_prompt_sections(
        definitions["system_message"],
        *explanation_sections,
        example_section,
        counterexample_section,
        f"{definitions['edge_task_fixed_sub_1']} {{formatted_subgraph1}}",
        f"{definitions['edge_task_fixed_sub_2']} {{formatted_subgraph2}}",
        f"{definitions['edge_task_fixed_sub_3']} {{expanded_label_context}}.",
        definitions["edge_conclusion_fixed"],
    )
    return {
        "label_expansion": label_expansion_prompt,
        "edge_suggestion": edge_suggestion_prompt,
        "cove_verification": definitions["cove_verification_template"],
    }


def _build_algo3_prompt_bundle(
    algorithm_config: Any,
    *,
    prompt_factors: dict[str, bool | int],
) -> dict[str, str]:
    definitions = algorithm_config.fragment_definitions
    if "system_message" not in definitions:
        tree_template = _select_template_name(
            algorithm_config,
            preferred="tree_expansion",
        )
        return {
            "tree_expansion": algorithm_config.assemble_prompt([], template_name=tree_template)
        }
    child_count = int(prompt_factors.get("child_count", 3))
    if child_count == 3:
        example_key = "example_3words"
        counterexample_key = "counter_example_3words"
    elif child_count == 5:
        example_key = "example_5words"
        counterexample_key = "counter_example_5words"
    else:
        raise ValueError(f"Unsupported ALGO3 child_count: {child_count}")
    example_section = (
        definitions[example_key] if bool(prompt_factors.get("include_example", False)) else ""
    )
    counterexample_section = (
        definitions[counterexample_key]
        if bool(prompt_factors.get("include_counterexample", False))
        else ""
    )
    tree_expansion_prompt = _join_prompt_sections(
        definitions["system_message"],
        definitions["explanation_variable"],
        f"{definitions['input_variable']} {{source_labels}}",
        definitions["task"].format(child_count, child_count),
        example_section,
        counterexample_section,
        definitions["conclusion"],
    )
    return {"tree_expansion": tree_expansion_prompt}


def _resolve_representation_variant(prompt_factors: dict[str, bool | int]) -> str:
    use_adjacency_notation = bool(prompt_factors.get("use_adjacency_notation", False))
    use_array_representation = bool(prompt_factors.get("use_array_representation", False))
    if use_adjacency_notation and use_array_representation:
        return "matrix"
    if use_adjacency_notation and not use_array_representation:
        return "markup"
    if not use_adjacency_notation and use_array_representation:
        return "edges"
    return "RDF"


def _join_prompt_sections(*sections: str) -> str:
    return " ".join(section.strip() for section in sections if section.strip())


def _select_template_name(algorithm_config: Any, *, preferred: str) -> str:
    if preferred in algorithm_config.prompt_templates:
        return preferred
    if "body" in algorithm_config.prompt_templates:
        return "body"
    return next(iter(algorithm_config.prompt_templates))


def render_prompt(template: str, **values: object) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{key}}}", str(value))
    return rendered


def generate_edges_from_prompt(chat_client: Any, prompt: str) -> list[Edge]:
    response = chat_client.complete_json(
        prompt=prompt,
        schema_name="edge_list",
        schema=edge_list_schema(),
    )
    raw_edges = cast(list[dict[str, object]], response["edges"])
    return [(str(edge["source"]), str(edge["target"])) for edge in raw_edges]


def verify_edges_from_prompt(
    chat_client: Any,
    prompt: str,
    candidate_edges: list[Edge],
) -> list[Edge]:
    resolved_prompt = (
        prompt if "{candidate_edges}" not in prompt else build_cove_prompt(candidate_edges)
    )
    response = chat_client.complete_json(
        prompt=resolved_prompt,
        schema_name="vote_list",
        schema=vote_list_schema(),
    )
    votes = [str(vote) for vote in cast(list[object], response["votes"])]
    return apply_cove_verification(candidate_edges, votes)


def propose_labels_from_prompt(chat_client: Any, prompt: str) -> list[str]:
    response = chat_client.complete_json(
        prompt=prompt,
        schema_name="label_list",
        schema=label_list_schema(),
    )
    return [str(label) for label in cast(list[object], response["labels"])]


def propose_children_from_prompt(chat_client: Any, prompt: str) -> dict[str, list[str]]:
    response = chat_client.complete_json(
        prompt=prompt,
        schema_name="children_by_label",
        schema=children_by_label_schema(),
    )
    raw_children = cast(dict[str, list[object]], response["children_by_label"])
    return {
        str(parent_label): [str(child_label) for child_label in child_labels]
        for parent_label, child_labels in raw_children.items()
    }


def edge_list_schema() -> dict[str, object]:
    return {
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
    }


def vote_list_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {"votes": {"type": "array", "items": {"type": "string"}}},
        "required": ["votes"],
        "additionalProperties": False,
    }


def label_list_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {"labels": {"type": "array", "items": {"type": "string"}}},
        "required": ["labels"],
        "additionalProperties": False,
    }


def children_by_label_schema() -> dict[str, object]:
    return {
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
