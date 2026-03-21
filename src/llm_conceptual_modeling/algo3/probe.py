import json
from dataclasses import dataclass
from pathlib import Path

from llm_conceptual_modeling.algo3.method import (
    build_tree_expander,
    execute_method3,
)
from llm_conceptual_modeling.algo3.mistral import (
    ChatCompletionClient,
    Method3PromptConfig,
    build_child_proposer,
    build_tree_expansion_prompt,
)
from llm_conceptual_modeling.algo3.tree import TreeExpansionNode
from llm_conceptual_modeling.post_revision_debug.artifacts import append_jsonl_event


@dataclass(frozen=True)
class Algo3ProbeSpec:
    run_name: str
    model: str
    source_labels: list[str]
    target_labels: list[str]
    prompt_config: Method3PromptConfig
    child_count: int
    max_depth: int
    output_dir: Path


def run_algo3_probe(
    *,
    spec: Algo3ProbeSpec,
    chat_client: ChatCompletionClient,
) -> dict[str, object]:
    output_dir = spec.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "summary.json"
    events_path = output_dir / "events.jsonl"
    tree_prompt_path = output_dir / "tree_expansion_prompt.txt"

    manifest_record = {
        "run_name": spec.run_name,
        "model": spec.model,
        "source_labels": spec.source_labels,
        "target_labels": spec.target_labels,
        "prompt_config": {
            "include_example": spec.prompt_config.include_example,
            "include_counterexample": spec.prompt_config.include_counterexample,
        },
        "child_count": spec.child_count,
        "max_depth": spec.max_depth,
    }
    manifest_path.write_text(json.dumps(manifest_record, indent=2))
    append_jsonl_event(events_path, {"event": "probe_started", **manifest_record})

    tree_prompt = build_tree_expansion_prompt(
        source_labels=spec.source_labels,
        child_count=spec.child_count,
        prompt_config=spec.prompt_config,
    )
    tree_prompt_path.write_text(tree_prompt)

    child_proposer = build_child_proposer(chat_client)
    tree_expander = build_tree_expander(child_proposer)
    execution_result = execute_method3(
        source_labels=spec.source_labels,
        target_labels=spec.target_labels,
        child_count=spec.child_count,
        max_depth=spec.max_depth,
        expand_tree=tree_expander,
    )

    summary_record = {
        "run_name": spec.run_name,
        "model": spec.model,
        "expanded_nodes": _nodes_to_json_compatible(execution_result.expanded_nodes),
        "matched_labels": execution_result.matched_labels,
    }
    summary_path.write_text(json.dumps(summary_record, indent=2))
    append_jsonl_event(events_path, {"event": "probe_finished", **summary_record})
    return summary_record


def _nodes_to_json_compatible(
    expanded_nodes: list[TreeExpansionNode],
) -> list[dict[str, object]]:
    node_records: list[dict[str, object]] = []

    for expanded_node in expanded_nodes:
        node_record = {
            "root_label": expanded_node.root_label,
            "parent_label": expanded_node.parent_label,
            "label": expanded_node.label,
            "depth": expanded_node.depth,
            "matched_target": expanded_node.matched_target,
        }
        node_records.append(node_record)

    return node_records
