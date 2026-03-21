import json
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

Edge = tuple[str, str]


@dataclass(frozen=True)
class Algo1ProbeSpec:
    run_name: str
    model: str
    subgraph1: list[Edge]
    subgraph2: list[Edge]
    prompt_config: Method1PromptConfig
    output_dir: Path


def run_algo1_probe(
    *,
    spec: Algo1ProbeSpec,
    chat_client: ChatCompletionClient,
) -> dict[str, object]:
    output_dir = spec.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "summary.json"
    events_path = output_dir / "events.jsonl"
    edge_prompt_path = output_dir / "edge_generation_prompt.txt"
    cove_prompt_path = output_dir / "cove_prompt.txt"

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
    manifest_path.write_text(json.dumps(manifest_record, indent=2))
    append_jsonl_event(events_path, {"event": "probe_started", **manifest_record})

    edge_prompt = build_direct_edge_prompt(
        subgraph1=spec.subgraph1,
        subgraph2=spec.subgraph2,
        prompt_config=spec.prompt_config,
    )
    edge_prompt_path.write_text(edge_prompt)

    edge_generator = build_edge_generator(chat_client)
    cove_verifier = build_cove_verifier(chat_client)
    execution_result = execute_method1(
        subgraph1=spec.subgraph1,
        subgraph2=spec.subgraph2,
        generate_edges=edge_generator,
        verify_edges=cove_verifier,
    )

    cove_prompt = build_cove_prompt(execution_result.candidate_edges)
    cove_prompt_path.write_text(cove_prompt)

    summary_record = {
        "run_name": spec.run_name,
        "model": spec.model,
        "candidate_edges": _edges_to_json_compatible(execution_result.candidate_edges),
        "verified_edges": _edges_to_json_compatible(execution_result.verified_edges),
    }
    summary_path.write_text(json.dumps(summary_record, indent=2))
    append_jsonl_event(events_path, {"event": "probe_finished", **summary_record})
    return summary_record


def _edges_to_json_compatible(edges: list[Edge]) -> list[list[str]]:
    edge_records: list[list[str]] = []

    for source, target in edges:
        edge_record = [source, target]
        edge_records.append(edge_record)

    return edge_records
