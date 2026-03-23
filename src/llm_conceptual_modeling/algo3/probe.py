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
from llm_conceptual_modeling.post_revision_debug.run_context import ProbeRunContext


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
    resume: bool = False


def run_algo3_probe(
    *,
    spec: Algo3ProbeSpec,
    chat_client: ChatCompletionClient,
) -> dict[str, object]:
    output_dir = spec.output_dir
    context = ProbeRunContext(
        output_dir=output_dir,
        run_name=spec.run_name,
        algorithm="algo3",
        resume=spec.resume,
    )

    if spec.resume:
        cached_summary = context.load_json("summary.json")
        if cached_summary is not None and context.is_stage_complete("probe_finished"):
            context.log("resume requested; returning cached summary", stage="resume")
            return cached_summary

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
    context.record_manifest(manifest_record)  # type: ignore[arg-type]
    context.append_event({"event": "probe_started", **manifest_record})

    tree_prompt = build_tree_expansion_prompt(
        source_labels=spec.source_labels,
        child_count=spec.child_count,
        prompt_config=spec.prompt_config,
    )
    if not context.is_stage_complete("tree_prompt_written"):
        context.record_prompt(
            "tree_expansion_prompt.txt",
            tree_prompt,
            stage="tree_prompt_written",
        )

    child_proposer = build_child_proposer(chat_client, spec.prompt_config)
    tree_expander = build_tree_expander(child_proposer)
    try:
        cached_execution = context.load_json("execution_checkpoint.json")
        if (
            spec.resume
            and cached_execution is not None
            and context.is_stage_complete("execution_completed")
        ):
            execution_result = cached_execution
        else:
            execution = execute_method3(
                source_labels=spec.source_labels,
                target_labels=spec.target_labels,
                child_count=spec.child_count,
                max_depth=spec.max_depth,
                expand_tree=tree_expander,
            )
            execution_result = {
                "expanded_nodes": _nodes_to_json_compatible(execution.expanded_nodes),
                "matched_labels": execution.matched_labels,
            }
            context.record_checkpoint(
                "execution_checkpoint.json",
                execution_result,  # type: ignore[arg-type]
                stage="execution_completed",
            )

        expanded_nodes: list[dict[str, object]] = execution_result["expanded_nodes"]  # type: ignore[index]
        matched_labels: list[str] = execution_result["matched_labels"]  # type: ignore[index]
        summary_record: dict[str, object] = {
            "run_name": spec.run_name,
            "model": spec.model,
            "expanded_nodes": expanded_nodes,
            "matched_labels": matched_labels,
        }
        context.record_checkpoint("summary.json", summary_record, stage="summary_written")
        context.mark_stage_complete("probe_finished", details={"summary_path": "summary.json"})
        context.log("probe finished", stage="complete")
        append_jsonl_event(context.events_path, {"event": "probe_finished", **summary_record})
        return summary_record
    except Exception as error:
        context.record_failure(error=error)
        append_jsonl_event(
            context.events_path,
            {
                "event": "probe_failed",
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )
        raise


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
        node_records.append(node_record)  # type: ignore[arg-type]

    return node_records
