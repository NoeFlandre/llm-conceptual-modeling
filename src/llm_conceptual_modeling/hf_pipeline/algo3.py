from __future__ import annotations

import json
from pathlib import Path

from llm_conceptual_modeling.algo3.method import (
    ChildDictionaryProposer,
    build_tree_expander,
    execute_method3,
)
from llm_conceptual_modeling.algo3.mistral import build_child_proposer
from llm_conceptual_modeling.common.hf_transformers import HFTransformersRuntimeFactory
from llm_conceptual_modeling.hf_batch.prompts import propose_children_from_prompt, render_prompt
from llm_conceptual_modeling.hf_batch.types import HFRunSpec, RuntimeResult
from llm_conceptual_modeling.hf_batch.utils import RecordingChatClient, runtime_details
from llm_conceptual_modeling.hf_batch_utils import algo3_prompt_config as _algo3_prompt_config
from llm_conceptual_modeling.hf_batch_utils import coerce_edges as _coerce_edges
from llm_conceptual_modeling.hf_batch_utils import collect_nodes as _collect_nodes
from llm_conceptual_modeling.hf_pipeline.common import (
    mark_worker_ready_for_execution as _mark_worker_ready_for_execution,
)
from llm_conceptual_modeling.hf_pipeline.metrics import (
    trace_metric_summary as _trace_metric_summary,
)


def run_algo3(
    spec: HFRunSpec,
    *,
    hf_runtime: HFTransformersRuntimeFactory,
    run_dir: Path | None = None,
) -> RuntimeResult:
    source_graph = _coerce_edges(spec.input_payload["source_graph"])
    target_graph = _coerce_edges(spec.input_payload["target_graph"])
    mother_graph = _coerce_edges(spec.input_payload["mother_graph"])
    source_labels = _collect_nodes(source_graph)
    target_labels = _collect_nodes(target_graph)
    prompt_factors = dict(spec.prompt_factors)
    child_count = int(prompt_factors.pop("child_count"))
    max_depth = int(prompt_factors.pop("max_depth"))
    prompt_config = _algo3_prompt_config(prompt_factors)
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
        result = execute_method3(
            source_labels=source_labels,
            target_labels=target_labels,
            child_count=child_count,
            max_depth=max_depth,
            expand_tree=build_tree_expander(
                build_child_proposer(recorder, prompt_config=prompt_config)
            ),
        )
    else:
        prompt_bundle = spec.prompt_bundle

        def configured_child_proposer(
            source_labels: list[str],
            *,
            child_count: int,
        ) -> dict[str, list[str]]:
            return propose_children_from_prompt(
                recorder,
                render_prompt(
                    prompt_bundle["tree_expansion"],
                    source_labels=repr(source_labels),
                    child_count=child_count,
                ),
            )

        child_proposer: ChildDictionaryProposer = configured_child_proposer
        result = execute_method3(
            source_labels=source_labels,
            target_labels=target_labels,
            child_count=child_count,
            max_depth=max_depth,
            expand_tree=build_tree_expander(child_proposer),
        )
    result_edges = [(node.parent_label, node.label) for node in result.expanded_nodes]
    raw_row = {
        **spec.raw_context,
        "Source Graph": repr(source_graph),
        "Target Graph": repr(target_graph),
        "Mother Graph": repr(mother_graph),
        "Results": repr(result_edges),
        "Recall": 0.0,
    }
    return {
        "raw_row": raw_row,
        "runtime": runtime_details(spec.runtime_profile),
        "raw_response": json.dumps(recorder.records, indent=2, sort_keys=True),
        "summary": {
            "result_edge_count": len(result_edges),
            "recall": float(raw_row["Recall"]),
            **_trace_metric_summary(recorder.records),
        },
    }
