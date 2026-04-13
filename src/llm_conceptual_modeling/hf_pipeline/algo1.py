from __future__ import annotations

import json
from pathlib import Path

from llm_conceptual_modeling.algo1.method import execute_method1
from llm_conceptual_modeling.algo1.mistral import build_cove_verifier, build_edge_generator
from llm_conceptual_modeling.common.hf_transformers import HFTransformersRuntimeFactory
from llm_conceptual_modeling.common.mistral import _format_knowledge_map
from llm_conceptual_modeling.hf_batch.prompts import (
    generate_edges_from_prompt,
    render_prompt,
    verify_edges_from_prompt,
)
from llm_conceptual_modeling.hf_batch.types import HFRunSpec, RuntimeResult
from llm_conceptual_modeling.hf_batch.utils import RecordingChatClient, runtime_details
from llm_conceptual_modeling.hf_batch.utils import write_json as _write_json
from llm_conceptual_modeling.hf_batch_utils import algo1_prompt_config as _algo1_prompt_config
from llm_conceptual_modeling.hf_batch_utils import coerce_edges as _coerce_edges
from llm_conceptual_modeling.hf_pipeline.common import (
    mark_worker_ready_for_execution as _mark_worker_ready_for_execution,
)
from llm_conceptual_modeling.hf_pipeline.metrics import (
    connection_metric_summary as _connection_metric_summary,
)
from llm_conceptual_modeling.hf_pipeline.metrics import (
    sanitize_algorithm_edge_result as _sanitize_algorithm_edge_result,
)
from llm_conceptual_modeling.hf_pipeline.metrics import (
    trace_metric_summary as _trace_metric_summary,
)


def run_algo1(
    spec: HFRunSpec,
    *,
    hf_runtime: HFTransformersRuntimeFactory,
    run_dir: Path | None = None,
) -> RuntimeResult:
    subgraph1 = _coerce_edges(spec.input_payload["subgraph1"])
    subgraph2 = _coerce_edges(spec.input_payload["subgraph2"])
    graph = _coerce_edges(spec.input_payload["graph"])
    prompt_config = _algo1_prompt_config(spec.prompt_factors)
    chat_client = hf_runtime.build_chat_client(
        model=spec.model,
        decoding_config=spec.decoding,
        max_new_tokens_by_schema=spec.max_new_tokens_by_schema,
        context_policy=spec.context_policy,
        seed=spec.seed,
    )
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
    if spec.prompt_bundle is None:
        result = execute_method1(
            subgraph1=subgraph1,
            subgraph2=subgraph2,
            generate_edges=build_edge_generator(recorder, prompt_config=prompt_config),
            verify_edges=build_cove_verifier(recorder),
        )
    else:
        prompt_bundle = spec.prompt_bundle
        formatted_subgraph1 = _format_knowledge_map(subgraph1, prompt_config=prompt_config)
        formatted_subgraph2 = _format_knowledge_map(subgraph2, prompt_config=prompt_config)
        stage_path = None if run_dir is None else run_dir / "stages" / "algo1_edge_generation.json"
        if stage_path is not None and stage_path.exists():
            stage_payload = json.loads(stage_path.read_text(encoding="utf-8"))
            candidate_edges = [tuple(edge) for edge in stage_payload["candidate_edges"]]
        else:
            candidate_edges = generate_edges_from_prompt(
                recorder,
                render_prompt(
                    prompt_bundle["direct_edge"],
                    formatted_subgraph1=formatted_subgraph1,
                    formatted_subgraph2=formatted_subgraph2,
                ),
            )
            if stage_path is not None:
                stage_path.parent.mkdir(parents=True, exist_ok=True)
                _write_json(
                    stage_path,
                    {"candidate_edges": [list(edge) for edge in candidate_edges]},
                )
        verified_edges = verify_edges_from_prompt(
            recorder,
            render_prompt(
                prompt_bundle["cove_verification"],
                candidate_edges=repr(candidate_edges),
            ),
            candidate_edges,
        )
        result = execute_method1(
            subgraph1=subgraph1,
            subgraph2=subgraph2,
            generate_edges=lambda *, subgraph1, subgraph2: candidate_edges,
            verify_edges=lambda candidate_edges: verified_edges,
        )
    sanitized_verified_edges = _sanitize_algorithm_edge_result("algo1", result.verified_edges)
    raw_row = {
        **spec.raw_context,
        "Result": repr(sanitized_verified_edges),
        "graph": repr(graph),
        "subgraph1": repr(subgraph1),
        "subgraph2": repr(subgraph2),
    }
    return {
        "raw_row": raw_row,
        "runtime": runtime_details(spec.runtime_profile),
        "raw_response": json.dumps(recorder.records, indent=2, sort_keys=True),
        "summary": {
            "candidate_edge_count": len(result.candidate_edges),
            "verified_edge_count": len(sanitized_verified_edges),
            **_trace_metric_summary(recorder.records),
            **_connection_metric_summary(raw_row),
        },
    }
