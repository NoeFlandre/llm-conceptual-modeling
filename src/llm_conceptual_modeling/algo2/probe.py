import json
from dataclasses import dataclass
from pathlib import Path

from llm_conceptual_modeling.algo2.embeddings import EmbeddingClient
from llm_conceptual_modeling.algo2.method import execute_method2
from llm_conceptual_modeling.algo2.mistral import (
    ChatCompletionClient,
    build_edge_suggester,
    build_edge_suggestion_prompt,
    build_label_expansion_prompt,
    build_label_proposer,
)
from llm_conceptual_modeling.common.graph_data import load_algo2_thesaurus
from llm_conceptual_modeling.post_revision_debug.artifacts import append_jsonl_event


@dataclass(frozen=True)
class Algo2ProbeSpec:
    run_name: str
    model: str
    seed_labels: list[str]
    convergence_threshold: float
    output_dir: Path


def run_algo2_probe(
    *,
    spec: Algo2ProbeSpec,
    chat_client: ChatCompletionClient,
    embedding_client: EmbeddingClient,
) -> dict[str, object]:
    output_dir = spec.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "summary.json"
    events_path = output_dir / "events.jsonl"
    label_prompt_path = output_dir / "label_expansion_prompt.txt"
    edge_prompt_path = output_dir / "edge_suggestion_prompt.txt"

    manifest_record = {
        "run_name": spec.run_name,
        "model": spec.model,
        "seed_labels": spec.seed_labels,
        "convergence_threshold": spec.convergence_threshold,
    }
    manifest_path.write_text(json.dumps(manifest_record, indent=2))
    append_jsonl_event(events_path, {"event": "probe_started", **manifest_record})

    label_prompt = build_label_expansion_prompt(spec.seed_labels)
    label_prompt_path.write_text(label_prompt)
    label_proposer = build_label_proposer(chat_client)
    edge_suggester = build_edge_suggester(chat_client)
    thesaurus = load_algo2_thesaurus()
    execution_result = execute_method2(
        seed_labels=spec.seed_labels,
        propose_labels=label_proposer,
        suggest_edges=edge_suggester,
        embedding_client=embedding_client,
        convergence_threshold=spec.convergence_threshold,
        thesaurus=thesaurus,
    )
    expanded_label_context = spec.seed_labels + execution_result.expanded_labels
    edge_prompt = build_edge_suggestion_prompt(expanded_label_context)
    edge_prompt_path.write_text(edge_prompt)
    raw_edges = _edges_to_json_compatible(execution_result.raw_edges)
    normalized_edges = _edges_to_json_compatible(execution_result.normalized_edges)

    summary_record = {
        "run_name": spec.run_name,
        "model": spec.model,
        "expanded_labels": execution_result.expanded_labels,
        "raw_edges": raw_edges,
        "normalized_edges": normalized_edges,
        "final_similarity": execution_result.final_similarity,
        "iteration_count": execution_result.iteration_count,
    }
    summary_path.write_text(json.dumps(summary_record, indent=2))
    append_jsonl_event(events_path, {"event": "probe_finished", **summary_record})
    return summary_record


def _edges_to_json_compatible(edges: list[tuple[str, str]]) -> list[list[str]]:
    edge_records: list[list[str]] = []

    for source, target in edges:
        edge_record = [source, target]
        edge_records.append(edge_record)

    return edge_records
