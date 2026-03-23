from dataclasses import dataclass
from pathlib import Path

from llm_conceptual_modeling.algo2.embeddings import EmbeddingClient
from llm_conceptual_modeling.algo2.method import execute_method2
from llm_conceptual_modeling.algo2.mistral import (
    ChatCompletionClient,
    Method2PromptConfig,
    build_edge_suggester,
    build_edge_suggestion_prompt,
    build_label_expansion_prompt,
    build_label_proposer,
)
from llm_conceptual_modeling.common.graph_data import load_algo2_thesaurus
from llm_conceptual_modeling.post_revision_debug.artifacts import append_jsonl_event
from llm_conceptual_modeling.post_revision_debug.run_context import ProbeRunContext


@dataclass(frozen=True)
class Algo2ProbeSpec:
    run_name: str
    model: str
    seed_labels: list[str]
    subgraph1: list[tuple[str, str]]
    subgraph2: list[tuple[str, str]]
    prompt_config: Method2PromptConfig
    convergence_threshold: float
    output_dir: Path
    resume: bool = False
    provider: str = "mistral"
    embedding_provider: str = "mistral"
    embedding_model: str = "mistral-embed-2312"


def run_algo2_probe(
    *,
    spec: Algo2ProbeSpec,
    chat_client: ChatCompletionClient,
    embedding_client: EmbeddingClient,
) -> dict[str, object]:
    output_dir = spec.output_dir
    context = ProbeRunContext(
        output_dir=output_dir,
        run_name=spec.run_name,
        algorithm="algo2",
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
        "provider": spec.provider,
        "embedding_provider": spec.embedding_provider,
        "embedding_model": spec.embedding_model,
        "seed_labels": spec.seed_labels,
        "subgraph1": _edges_to_json_compatible(spec.subgraph1),
        "subgraph2": _edges_to_json_compatible(spec.subgraph2),
        "prompt_config": {
            "use_adjacency_notation": spec.prompt_config.use_adjacency_notation,
            "use_array_representation": spec.prompt_config.use_array_representation,
            "include_explanation": spec.prompt_config.include_explanation,
            "include_example": spec.prompt_config.include_example,
            "include_counterexample": spec.prompt_config.include_counterexample,
            "use_relaxed_convergence": spec.prompt_config.use_relaxed_convergence,
        },
        "convergence_threshold": spec.convergence_threshold,
    }
    context.record_manifest(manifest_record)  # type: ignore[arg-type]
    context.append_event({"event": "probe_started", **manifest_record})

    label_prompt = build_label_expansion_prompt(
        spec.seed_labels,
        subgraph1=spec.subgraph1,
        subgraph2=spec.subgraph2,
        prompt_config=spec.prompt_config,
    )
    if not context.is_stage_complete("label_prompt_written"):
        context.record_prompt(
            "label_expansion_prompt.txt",
            label_prompt,
            stage="label_prompt_written",
        )
    label_proposer = build_label_proposer(chat_client, spec.prompt_config)
    edge_suggester = build_edge_suggester(chat_client, spec.prompt_config)
    thesaurus = load_algo2_thesaurus()
    try:
        cached_execution = context.load_json("execution_checkpoint.json")
        if (
            spec.resume
            and cached_execution is not None
            and context.is_stage_complete("execution_completed")
        ):
            execution_result = cached_execution
        else:
            execution = execute_method2(
                seed_labels=spec.seed_labels,
                existing_edges=spec.subgraph1 + spec.subgraph2,
                propose_labels=label_proposer,
                suggest_edges=edge_suggester,
                embedding_client=embedding_client,
                convergence_threshold=spec.convergence_threshold,
                thesaurus=thesaurus,
            )
            execution_result = {
                "expanded_labels": execution.expanded_labels,
                "raw_edges": _edges_to_json_compatible(execution.raw_edges),
                "normalized_edges": _edges_to_json_compatible(execution.normalized_edges),
                "final_similarity": execution.final_similarity,
                "iteration_count": execution.iteration_count,
            }
            context.record_checkpoint(
                "execution_checkpoint.json",
                execution_result,  # type: ignore[arg-type]
                stage="execution_completed",
            )
        expanded_labels: list[str] = execution_result["expanded_labels"]  # type: ignore[index]
        expanded_label_context = spec.seed_labels + expanded_labels
        edge_prompt = build_edge_suggestion_prompt(
            expanded_label_context,
            subgraph1=spec.subgraph1,
            subgraph2=spec.subgraph2,
            prompt_config=spec.prompt_config,
        )
        if not context.is_stage_complete("edge_prompt_written"):
            context.record_prompt(
                "edge_suggestion_prompt.txt",
                edge_prompt,
                stage="edge_prompt_written",
            )

        summary_record = {
            "run_name": spec.run_name,
            "model": spec.model,
            "provider": spec.provider,
            "embedding_provider": spec.embedding_provider,
            "embedding_model": spec.embedding_model,
            "expanded_labels": execution_result["expanded_labels"],
            "raw_edges": execution_result["raw_edges"],
            "normalized_edges": execution_result["normalized_edges"],
            "final_similarity": execution_result["final_similarity"],
            "iteration_count": execution_result["iteration_count"],
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


def _edges_to_json_compatible(edges: list[tuple[str, str]]) -> list[list[str]]:
    edge_records: list[list[str]] = []

    for source, target in edges:
        edge_record = [source, target]
        edge_records.append(edge_record)

    return edge_records
