from dataclasses import dataclass
from pathlib import Path

from llm_conceptual_modeling.algo1.cove import build_cove_prompt
from llm_conceptual_modeling.algo1.method import execute_method1
from llm_conceptual_modeling.algo1.mistral import (
    ChatCompletionClient,
    Method1PromptConfig,
    build_cove_verifier,
    build_direct_edge_prompt,
    build_edge_generator,
)
from llm_conceptual_modeling.post_revision_debug.artifacts import append_jsonl_event
from llm_conceptual_modeling.post_revision_debug.run_context import ProbeRunContext

Edge = tuple[str, str]


@dataclass(frozen=True)
class Algo1ProbeSpec:
    run_name: str
    model: str
    subgraph1: list[Edge]
    subgraph2: list[Edge]
    prompt_config: Method1PromptConfig
    output_dir: Path
    resume: bool = False


def run_algo1_probe(
    *,
    spec: Algo1ProbeSpec,
    chat_client: ChatCompletionClient,
) -> dict[str, object]:
    output_dir = spec.output_dir
    context = ProbeRunContext(
        output_dir=output_dir,
        run_name=spec.run_name,
        algorithm="algo1",
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
        "subgraph1": _edges_to_json_compatible(spec.subgraph1),
        "subgraph2": _edges_to_json_compatible(spec.subgraph2),
        "prompt_config": {
            "use_adjacency_notation": spec.prompt_config.use_adjacency_notation,
            "use_array_representation": spec.prompt_config.use_array_representation,
            "include_explanation": spec.prompt_config.include_explanation,
            "include_example": spec.prompt_config.include_example,
            "include_counterexample": spec.prompt_config.include_counterexample,
        },
    }
    context.record_manifest(manifest_record)
    context.append_event({"event": "probe_started", **manifest_record})

    edge_prompt = build_direct_edge_prompt(
        subgraph1=spec.subgraph1,
        subgraph2=spec.subgraph2,
        prompt_config=spec.prompt_config,
    )
    if not context.is_stage_complete("edge_prompt_written"):
        context.record_prompt(
            "edge_generation_prompt.txt",
            edge_prompt,
            stage="edge_prompt_written",
        )

    edge_generator = build_edge_generator(chat_client)
    cove_verifier = build_cove_verifier(chat_client)
    try:
        cached_execution = context.load_json("execution_checkpoint.json")
        if spec.resume and cached_execution is not None and context.is_stage_complete(
            "execution_completed"
        ):
            execution_result = cached_execution
        else:
            execution = execute_method1(
                subgraph1=spec.subgraph1,
                subgraph2=spec.subgraph2,
                generate_edges=edge_generator,
                verify_edges=cove_verifier,
            )
            execution_result = {
                "candidate_edges": _edges_to_json_compatible(execution.candidate_edges),
                "verified_edges": _edges_to_json_compatible(execution.verified_edges),
            }
            context.record_checkpoint(
                "execution_checkpoint.json",
                execution_result,
                stage="execution_completed",
            )

        cove_prompt = build_cove_prompt(
            [tuple(edge) for edge in execution_result["candidate_edges"]],
        )
        if not context.is_stage_complete("cove_prompt_written"):
            context.record_prompt(
                "cove_prompt.txt",
                cove_prompt,
                stage="cove_prompt_written",
            )

        summary_record = {
            "run_name": spec.run_name,
            "model": spec.model,
            "candidate_edges": execution_result["candidate_edges"],
            "verified_edges": execution_result["verified_edges"],
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


def _edges_to_json_compatible(edges: list[Edge]) -> list[list[str]]:
    edge_records: list[list[str]] = []

    for edge in edges:
        if isinstance(edge, (list, tuple)):
            edge_record = [str(value) for value in edge]
            edge_records.append(edge_record)
            continue

        edge_record = [str(edge)]
        edge_records.append(edge_record)

    return edge_records
