from __future__ import annotations

import json
from pathlib import Path

from llm_conceptual_modeling.algo1.mistral import build_cove_verifier
from llm_conceptual_modeling.algo2.method import execute_method2
from llm_conceptual_modeling.algo2.mistral import build_edge_suggester, build_label_proposer
from llm_conceptual_modeling.common.graph_data import load_algo2_thesaurus
from llm_conceptual_modeling.common.hf_transformers import HFTransformersRuntimeFactory
from llm_conceptual_modeling.common.mistral import _format_knowledge_map
from llm_conceptual_modeling.hf_batch.prompts import (
    generate_edges_from_prompt,
    propose_labels_from_prompt,
    render_prompt,
    verify_edges_from_prompt,
)
from llm_conceptual_modeling.hf_batch.types import HFRunSpec, RuntimeResult
from llm_conceptual_modeling.hf_batch.utils import RecordingChatClient, runtime_details
from llm_conceptual_modeling.hf_batch.utils import write_json as _write_json
from llm_conceptual_modeling.hf_batch.utils import algo2_prompt_config as _algo2_prompt_config
from llm_conceptual_modeling.hf_batch.utils import coerce_edges as _coerce_edges
from llm_conceptual_modeling.hf_batch.utils import collect_nodes as _collect_nodes
from llm_conceptual_modeling.hf_pipeline.common import (
    mark_worker_ready_for_execution as _mark_worker_ready_for_execution,
)
from llm_conceptual_modeling.hf_pipeline.metrics import (
    connection_metric_summary as _connection_metric_summary,
)
from llm_conceptual_modeling.hf_pipeline.metrics import (
    trace_metric_summary as _trace_metric_summary,
)


def run_algo2(
    spec: HFRunSpec,
    *,
    hf_runtime: HFTransformersRuntimeFactory,
    run_dir: Path | None = None,
) -> RuntimeResult:
    subgraph1 = _coerce_edges(spec.input_payload["subgraph1"])
    subgraph2 = _coerce_edges(spec.input_payload["subgraph2"])
    graph = _coerce_edges(spec.input_payload["graph"])
    prompt_config = _algo2_prompt_config(spec.prompt_factors)
    chat_client = hf_runtime.build_chat_client(
        model=spec.model,
        decoding_config=spec.decoding,
        max_new_tokens_by_schema=spec.max_new_tokens_by_schema,
        context_policy=spec.context_policy,
        seed=spec.seed,
    )
    embedding_client = hf_runtime.build_embedding_client(model=spec.embedding_model)
    _mark_worker_ready_for_execution(run_dir)
    recorder = RecordingChatClient(
        chat_client,
        persist_path=(run_dir / "raw_response.json") if run_dir is not None else None,
        active_stage_path=(run_dir / "active_stage.json") if run_dir is not None else None,
        active_stage_context={
            "algorithm": spec.algorithm,
            "pair_name": spec.pair_name,
            "condition_bits": spec.condition_bits,
            "replication": spec.replication,
            "model": spec.model,
            "decoding_algorithm": spec.decoding.algorithm,
        },
    )
    source_labels = _collect_nodes(subgraph1)
    target_labels = _collect_nodes(subgraph2)
    seed_labels = source_labels + [label for label in target_labels if label not in source_labels]
    threshold = 0.02 if prompt_config.use_relaxed_convergence else 0.01
    thesaurus = load_algo2_thesaurus()
    if spec.prompt_bundle is None:
        result = execute_method2(
            seed_labels=seed_labels,
            existing_edges=list(subgraph1) + list(subgraph2),
            propose_labels=build_label_proposer(recorder, prompt_config=prompt_config),
            suggest_edges=build_edge_suggester(recorder, prompt_config=prompt_config),
            verify_edges=build_cove_verifier(recorder),
            embedding_client=embedding_client,
            convergence_threshold=threshold,
            thesaurus=thesaurus,
        )
    else:
        prompt_bundle = spec.prompt_bundle
        formatted_subgraph1 = _format_knowledge_map(subgraph1, prompt_config=prompt_config)
        formatted_subgraph2 = _format_knowledge_map(subgraph2, prompt_config=prompt_config)
        result = execute_method2(
            seed_labels=seed_labels,
            existing_edges=list(subgraph1) + list(subgraph2),
            propose_labels=lambda current_labels: propose_labels_from_prompt(
                recorder,
                render_prompt(
                    prompt_bundle["label_expansion"],
                    formatted_subgraph1=formatted_subgraph1,
                    formatted_subgraph2=formatted_subgraph2,
                    seed_labels=", ".join(current_labels),
                ),
            ),
            suggest_edges=lambda expanded_label_context: generate_edges_from_prompt(
                recorder,
                render_prompt(
                    prompt_bundle["edge_suggestion"],
                    formatted_subgraph1=formatted_subgraph1,
                    formatted_subgraph2=formatted_subgraph2,
                    expanded_label_context=", ".join(expanded_label_context),
                ),
            ),
            verify_edges=lambda candidate_edges: verify_edges_from_prompt(
                recorder,
                render_prompt(
                    prompt_bundle["cove_verification"],
                    candidate_edges=repr(candidate_edges),
                ),
                candidate_edges,
            ),
            embedding_client=embedding_client,
            convergence_threshold=threshold,
            thesaurus=thesaurus,
        )
    if run_dir is not None:
        stages_dir = run_dir / "stages"
        stages_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            stages_dir / "algo2_label_expansion.json",
            {
                "expanded_labels": result.expanded_labels,
                "final_similarity": result.final_similarity,
                "iteration_count": result.iteration_count,
            },
        )
        _write_json(
            stages_dir / "algo2_edge_generation.json",
            {
                "raw_edges": [list(edge) for edge in result.raw_edges],
                "verified_edges": [list(edge) for edge in result.normalized_edges],
            },
        )
    raw_row = {
        **spec.raw_context,
        "Result": repr(result.normalized_edges),
        "graph": repr(graph),
        "subgraph1": repr(subgraph1),
        "subgraph2": repr(subgraph2),
    }
    return {
        "raw_row": raw_row,
        "runtime": runtime_details(spec.runtime_profile),
        "raw_response": json.dumps(recorder.records, indent=2, sort_keys=True),
        "summary": {
            "candidate_edge_count": len(result.raw_edges),
            "verified_edge_count": len(result.normalized_edges),
            **_trace_metric_summary(recorder.records),
            **_connection_metric_summary(raw_row),
        },
    }
