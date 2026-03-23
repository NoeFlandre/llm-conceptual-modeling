from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from llm_conceptual_modeling.algo1.cove import build_cove_prompt
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


class _LoggedChatClient:
    def __init__(self, *, inner: ChatCompletionClient, context: ProbeRunContext) -> None:
        self._inner = inner
        self._context = context
        self._call_index = 0

    def complete_json(
        self,
        *,
        prompt: str,
        schema_name: str,
        schema: dict[str, object],
    ) -> dict[str, object]:
        self._call_index += 1
        call_index = self._call_index
        prompt_chars = len(prompt)
        self._context.append_event(
            {
                "event": "chat_call_started",
                "call_index": call_index,
                "schema_name": schema_name,
                "prompt_chars": prompt_chars,
            }
        )
        self._context.log(
            f"chat call started schema={schema_name} prompt_chars={prompt_chars}",
            stage="chat_call_started",
        )
        try:
            response = self._inner.complete_json(
                prompt=prompt,
                schema_name=schema_name,
                schema=schema,
            )
        except Exception as error:
            self._context.append_event(
                {
                    "event": "chat_call_failed",
                    "call_index": call_index,
                    "schema_name": schema_name,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
            self._context.log(
                f"chat call failed schema={schema_name} error={type(error).__name__}",
                level="ERROR",
                stage="chat_call_failed",
            )
            raise

        self._context.append_event(
            {
                "event": "chat_call_completed",
                "call_index": call_index,
                "schema_name": schema_name,
            }
        )
        self._context.log(
            f"chat call completed schema={schema_name}",
            stage="chat_call_completed",
        )
        return response


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
    context.record_manifest(manifest_record)  # type: ignore[arg-type]
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

    logged_chat_client = _LoggedChatClient(inner=chat_client, context=context)
    edge_generator = build_edge_generator(logged_chat_client, spec.prompt_config)
    cove_verifier = build_cove_verifier(logged_chat_client)
    try:
        candidate_edges = _load_or_generate_candidate_edges(
            context=context,
            resume=spec.resume,
            edge_generator=edge_generator,
            subgraph1=spec.subgraph1,
            subgraph2=spec.subgraph2,
        )
        verified_edges = _load_or_generate_verified_edges(
            context=context,
            resume=spec.resume,
            cove_verifier=cove_verifier,
            candidate_edges=candidate_edges,
        )
        cove_prompt = build_cove_prompt(
            [tuple(edge) for edge in candidate_edges],  # type: ignore[arg-type]
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
            "candidate_edges": candidate_edges,
            "verified_edges": verified_edges,
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


def _load_or_generate_candidate_edges(
    *,
    context: ProbeRunContext,
    resume: bool,
    edge_generator: object,
    subgraph1: list[Edge],
    subgraph2: list[Edge],
) -> list[list[str]]:
    cached_candidate = context.load_json("candidate_edges.json")
    if (
        resume
        and cached_candidate is not None
        and context.is_stage_complete("candidate_edges_completed")
    ):
        candidate_edges = cached_candidate.get("candidate_edges")
        if isinstance(candidate_edges, list):
            return _normalize_json_edge_records(candidate_edges)

    generated_candidate_edges = edge_generator(subgraph1=subgraph1, subgraph2=subgraph2)  # type: ignore[call-arg]
    candidate_edges = _edges_to_json_compatible(generated_candidate_edges)
    context.record_checkpoint(
        "candidate_edges.json",
        {"candidate_edges": candidate_edges},
        stage="candidate_edges_completed",
    )
    return candidate_edges


def _load_or_generate_verified_edges(
    *,
    context: ProbeRunContext,
    resume: bool,
    cove_verifier: object,
    candidate_edges: list[list[str]],
) -> list[list[str]]:
    cached_execution = context.load_json("execution_checkpoint.json")
    if resume and cached_execution is not None and context.is_stage_complete("execution_completed"):
        verified_edges = cached_execution.get("verified_edges")
        if isinstance(verified_edges, list):
            return _normalize_json_edge_records(verified_edges)

    verified_edges = cove_verifier([tuple(edge) for edge in candidate_edges])  # type: ignore[call-arg]
    verified_edge_records = _edges_to_json_compatible(verified_edges)
    context.record_checkpoint(
        "execution_checkpoint.json",
        {
            "candidate_edges": candidate_edges,
            "verified_edges": verified_edge_records,
        },
        stage="execution_completed",
    )
    return verified_edge_records


def _normalize_json_edge_records(edges: list[object]) -> list[list[str]]:
    normalized_edges: list[list[str]] = []
    for edge in edges:
        if isinstance(edge, (list, tuple)):
            normalized_edges.append([str(value) for value in edge])
    return normalized_edges


def _edges_to_json_compatible(edges: Sequence[str | Edge]) -> list[list[str]]:
    edge_records: list[list[str]] = []

    for edge in edges:
        if isinstance(edge, (list, tuple)):
            edge_record: list[str] = list(str(value) for value in edge)
            edge_records.append(edge_record)
            continue

        edge_record: list[str] = [str(edge)]
        edge_records.append(edge_record)

    return edge_records
