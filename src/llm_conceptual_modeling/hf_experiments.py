from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd

from llm_conceptual_modeling.algo1.method import execute_method1
from llm_conceptual_modeling.algo1.mistral import (
    build_cove_verifier,
    build_edge_generator,
)
from llm_conceptual_modeling.algo2.method import execute_method2
from llm_conceptual_modeling.algo2.mistral import (
    build_edge_suggester,
    build_label_proposer,
)
from llm_conceptual_modeling.algo3.method import (
    ChildDictionaryProposer,
    build_tree_expander,
    execute_method3,
)
from llm_conceptual_modeling.algo3.mistral import build_child_proposer
from llm_conceptual_modeling.common.graph_data import load_algo2_thesaurus
from llm_conceptual_modeling.common.hf_transformers import (
    HFTransformersRuntimeFactory,
    RuntimeProfile,
    build_runtime_factory,
)
from llm_conceptual_modeling.common.mistral import _format_knowledge_map
from llm_conceptual_modeling.hf_batch.monitoring import (
    current_run_payload as _current_run_payload,
)
from llm_conceptual_modeling.hf_batch.monitoring import (
    status_timestamp_now as _status_timestamp_now,
)
from llm_conceptual_modeling.hf_batch.monitoring import (
    write_status_snapshot as _write_status_snapshot,
)
from llm_conceptual_modeling.hf_batch.outputs import write_aggregated_outputs
from llm_conceptual_modeling.hf_batch_planning import plan_paper_batch_specs
from llm_conceptual_modeling.hf_batch_prompts import (
    build_prompt_bundle as _build_prompt_bundle,  # noqa: F401
)
from llm_conceptual_modeling.hf_batch_prompts import (
    generate_edges_from_prompt as _generate_edges_from_prompt,
)
from llm_conceptual_modeling.hf_batch_prompts import (
    propose_children_from_prompt as _propose_children_from_prompt,
)
from llm_conceptual_modeling.hf_batch_prompts import (
    propose_labels_from_prompt as _propose_labels_from_prompt,
)
from llm_conceptual_modeling.hf_batch_prompts import (
    render_prompt as _render_prompt,
)
from llm_conceptual_modeling.hf_batch_prompts import (
    verify_edges_from_prompt as _verify_edges_from_prompt,
)
from llm_conceptual_modeling.hf_batch_types import HFRunSpec, RuntimeFactory, RuntimeResult
from llm_conceptual_modeling.hf_batch_utils import (
    RecordingChatClient as _RecordingChatClient,
)
from llm_conceptual_modeling.hf_batch_utils import (
    algo1_prompt_config as _algo1_prompt_config,
)
from llm_conceptual_modeling.hf_batch_utils import (
    algo2_prompt_config as _algo2_prompt_config,
)
from llm_conceptual_modeling.hf_batch_utils import (
    algo3_prompt_config as _algo3_prompt_config,
)
from llm_conceptual_modeling.hf_batch_utils import (
    coerce_edges as _coerce_edges,
)
from llm_conceptual_modeling.hf_batch_utils import (
    collect_nodes as _collect_nodes,
)
from llm_conceptual_modeling.hf_batch_utils import (
    manifest_for_spec as _manifest_for_spec,
)
from llm_conceptual_modeling.hf_batch_utils import (
    resolve_hf_token as _resolve_hf_token,
)
from llm_conceptual_modeling.hf_batch_utils import (
    runtime_details as _runtime_details,
)
from llm_conceptual_modeling.hf_batch_utils import (
    slugify_model as _slugify_model,
)
from llm_conceptual_modeling.hf_batch_utils import (
    write_json as _write_json,
)
from llm_conceptual_modeling.hf_batch_utils import (
    write_text as _write_text,
)
from llm_conceptual_modeling.hf_run_config import HFRunConfig


def plan_paper_batch(
    *,
    models: list[str],
    embedding_model: str,
    replications: int,
    algorithms: tuple[str, ...] | None = None,
    config: HFRunConfig | None = None,
    runtime_profile_provider: Callable[[str], RuntimeProfile] | None = None,
) -> list[HFRunSpec]:
    return plan_paper_batch_specs(
        models=models,
        embedding_model=embedding_model,
        replications=replications,
        algorithms=algorithms,
        config=config,
        runtime_profile_provider=runtime_profile_provider,
    )


def run_paper_batch(
    *,
    output_root: str | Path,
    models: list[str],
    embedding_model: str,
    replications: int,
    algorithms: tuple[str, ...] | None = None,
    config: HFRunConfig | None = None,
    runtime_factory: RuntimeFactory | None = None,
    resume: bool = False,
    dry_run: bool = False,
) -> None:
    if config is not None:
        output_root = config.run.output_root
        models = config.models.chat_models
        embedding_model = config.models.embedding_model
        replications = config.run.replications
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    hf_runtime = None if runtime_factory is not None else build_runtime_factory(
        hf_token=_resolve_hf_token()
    )
    if dry_run:
        profile_provider = None
    else:
        profile_provider = hf_runtime.profile_for_chat_model if hf_runtime else None
    planned_specs = plan_paper_batch(
        models=models,
        embedding_model=embedding_model,
        replications=replications,
        algorithms=algorithms,
        config=config,
        runtime_profile_provider=profile_provider,
    )
    if runtime_factory is None:
        if hf_runtime is None:
            raise ValueError("Missing HF runtime for non-dry local execution.")
        runtime_factory = _runtime_factory_from_hf_runtime(hf_runtime)

    summary_rows: list[dict[str, object]] = []
    total_runs = len(planned_specs)
    started_at = _status_timestamp_now()
    last_completed_run: dict[str, object] | None = None
    status_snapshot: dict[str, object] = {
        "total_runs": total_runs,
        "finished_count": 0,
        "failed_count": 0,
        "running_count": 0,
        "pending_count": total_runs,
        "failure_count": 0,
        "failures": [],
        "percent_complete": 0.0,
        "current_run": None,
        "last_completed_run": None,
        "started_at": started_at,
        "updated_at": started_at,
    }
    _write_status_snapshot(output_root=output_root_path, status=status_snapshot)

    for spec in planned_specs:
        run_dir = (
            output_root_path
            / "runs"
            / spec.algorithm
            / _slugify_model(spec.model)
            / spec.condition_label
            / spec.pair_name
            / spec.condition_bits
            / f"rep_{spec.replication:02d}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_dir / "summary.json"
        raw_row_path = run_dir / "raw_row.json"

        if resume and _is_finished_run_directory(run_dir):
            cached = json.loads(summary_path.read_text(encoding="utf-8"))
            summary_rows.append(cached)
            last_completed_run = _current_run_payload(
                algorithm=spec.algorithm,
                model=spec.model,
                decoding_algorithm=spec.decoding.algorithm,
                pair_name=spec.pair_name,
                condition_bits=spec.condition_bits,
                replication=spec.replication,
            )
            status_snapshot["finished_count"] = int(status_snapshot["finished_count"]) + 1
            status_snapshot["pending_count"] = int(status_snapshot["pending_count"]) - 1
            status_snapshot["last_completed_run"] = last_completed_run
            status_snapshot["percent_complete"] = round(
                (int(status_snapshot["finished_count"]) / total_runs) * 100.0,
                2,
            )
            status_snapshot["updated_at"] = _status_timestamp_now()
            _write_status_snapshot(output_root=output_root_path, status=status_snapshot)
            continue

        _write_json(run_dir / "manifest.json", _manifest_for_spec(spec))
        _write_json(run_dir / "state.json", {"status": "running"})
        current_run = _current_run_payload(
            algorithm=spec.algorithm,
            model=spec.model,
            decoding_algorithm=spec.decoding.algorithm,
            pair_name=spec.pair_name,
            condition_bits=spec.condition_bits,
            replication=spec.replication,
        )
        status_snapshot["current_run"] = current_run
        status_snapshot["running_count"] = 1
        status_snapshot["updated_at"] = _status_timestamp_now()
        _write_status_snapshot(output_root=output_root_path, status=status_snapshot)

        try:
            runtime_result = _execute_run(
                spec=spec,
                runtime_factory=runtime_factory,
                dry_run=dry_run,
            )
        except Exception as error:
            _write_json(
                run_dir / "error.json",
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "status": "failed",
                },
            )
            _write_json(run_dir / "state.json", {"status": "failed"})
            status_snapshot["running_count"] = 0
            status_snapshot["current_run"] = None
            status_snapshot["failed_count"] = int(status_snapshot["failed_count"]) + 1
            status_snapshot["pending_count"] = int(status_snapshot["pending_count"]) - 1
            failures = list(status_snapshot["failures"])
            failures.append(
                {
                    "run_dir": str(run_dir),
                    "message": str(error),
                    "type": type(error).__name__,
                }
            )
            status_snapshot["failures"] = failures
            status_snapshot["failure_count"] = len(failures)
            status_snapshot["updated_at"] = _status_timestamp_now()
            _write_status_snapshot(output_root=output_root_path, status=status_snapshot)
            raise

        raw_row = runtime_result["raw_row"]
        _write_json(raw_row_path, raw_row)
        _write_json(run_dir / "state.json", {"status": "finished"})
        _write_json(run_dir / "runtime.json", runtime_result["runtime"])
        _write_text(run_dir / "raw_response.json", runtime_result["raw_response"])

        summary = {
            "algorithm": spec.algorithm,
            "model": spec.model,
            "embedding_model": spec.embedding_model,
            "decoding_algorithm": spec.decoding.algorithm,
            "condition_label": spec.condition_label,
            "pair_name": spec.pair_name,
            "condition_bits": spec.condition_bits,
            "replication": spec.replication,
            "status": "finished",
            "thinking_mode_supported": runtime_result["runtime"].get(
                "thinking_mode_supported", False
            ),
            "raw_row_path": str(raw_row_path),
        }
        _write_json(summary_path, summary)
        summary_rows.append(summary)
        last_completed_run = current_run
        status_snapshot["running_count"] = 0
        status_snapshot["current_run"] = None
        status_snapshot["last_completed_run"] = last_completed_run
        status_snapshot["finished_count"] = int(status_snapshot["finished_count"]) + 1
        status_snapshot["pending_count"] = int(status_snapshot["pending_count"]) - 1
        status_snapshot["percent_complete"] = round(
            (int(status_snapshot["finished_count"]) / total_runs) * 100.0,
            2,
        )
        status_snapshot["updated_at"] = _status_timestamp_now()
        _write_status_snapshot(output_root=output_root_path, status=status_snapshot)

    summary_frame = pd.DataFrame.from_records(summary_rows)
    summary_frame.to_csv(output_root_path / "batch_summary.csv", index=False)
    if dry_run:
        return

    write_aggregated_outputs(output_root_path, summary_frame)


def _is_finished_run_directory(run_dir: Path) -> bool:
    state_path = run_dir / "state.json"
    summary_path = run_dir / "summary.json"
    required_paths = [
        run_dir / "manifest.json",
        state_path,
        run_dir / "runtime.json",
        run_dir / "raw_response.json",
        run_dir / "raw_row.json",
        summary_path,
    ]
    if not all(path.exists() for path in required_paths):
        return False
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "finished":
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary.get("status") == "finished"


def _execute_run(
    *,
    spec: HFRunSpec,
    runtime_factory: RuntimeFactory,
    dry_run: bool,
) -> RuntimeResult:
    if dry_run:
        return {
            "raw_row": dict(spec.raw_context),
            "runtime": {
                "thinking_mode_supported": spec.runtime_profile.supports_thinking_toggle,
                "device": spec.runtime_profile.device,
                "dtype": spec.runtime_profile.dtype,
                "quantization": spec.runtime_profile.quantization,
            },
            "raw_response": "[]",
        }
    return runtime_factory(spec)


def _runtime_factory_from_hf_runtime(hf_runtime: HFTransformersRuntimeFactory) -> RuntimeFactory:
    def runtime(spec: HFRunSpec) -> RuntimeResult:
        if spec.algorithm == "algo1":
            return _run_algo1(spec, hf_runtime=hf_runtime)
        if spec.algorithm == "algo2":
            return _run_algo2(spec, hf_runtime=hf_runtime)
        if spec.algorithm == "algo3":
            return _run_algo3(spec, hf_runtime=hf_runtime)
        raise ValueError(f"Unsupported algorithm: {spec.algorithm}")

    return runtime


def _run_algo1(
    spec: HFRunSpec,
    *,
    hf_runtime: HFTransformersRuntimeFactory,
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
    recorder = _RecordingChatClient(chat_client)
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
        result = execute_method1(
            subgraph1=subgraph1,
            subgraph2=subgraph2,
            generate_edges=lambda *, subgraph1, subgraph2: _generate_edges_from_prompt(
                recorder,
                _render_prompt(
                    prompt_bundle["direct_edge"],
                    formatted_subgraph1=formatted_subgraph1,
                    formatted_subgraph2=formatted_subgraph2,
                ),
            ),
            verify_edges=lambda candidate_edges: _verify_edges_from_prompt(
                recorder,
                _render_prompt(
                    prompt_bundle["cove_verification"],
                    candidate_edges=repr(candidate_edges),
                ),
                candidate_edges,
            ),
        )
    raw_row = {
        **spec.raw_context,
        "Result": repr(result.verified_edges),
        "graph": repr(graph),
        "subgraph1": repr(subgraph1),
        "subgraph2": repr(subgraph2),
    }
    return {
        "raw_row": raw_row,
        "runtime": _runtime_details(spec.runtime_profile),
        "raw_response": json.dumps(recorder.records, indent=2, sort_keys=True),
    }


def _run_algo2(
    spec: HFRunSpec,
    *,
    hf_runtime: HFTransformersRuntimeFactory,
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
    recorder = _RecordingChatClient(chat_client)
    embedding_client = hf_runtime.build_embedding_client(model=spec.embedding_model)
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
            propose_labels=lambda current_labels: _propose_labels_from_prompt(
                recorder,
                _render_prompt(
                    prompt_bundle["label_expansion"],
                    formatted_subgraph1=formatted_subgraph1,
                    formatted_subgraph2=formatted_subgraph2,
                    seed_labels=", ".join(current_labels),
                ),
            ),
            suggest_edges=lambda expanded_label_context: _generate_edges_from_prompt(
                recorder,
                _render_prompt(
                    prompt_bundle["edge_suggestion"],
                    formatted_subgraph1=formatted_subgraph1,
                    formatted_subgraph2=formatted_subgraph2,
                    expanded_label_context=", ".join(expanded_label_context),
                ),
            ),
            verify_edges=lambda candidate_edges: _verify_edges_from_prompt(
                recorder,
                _render_prompt(
                    prompt_bundle["cove_verification"],
                    candidate_edges=repr(candidate_edges),
                ),
                candidate_edges,
            ),
            embedding_client=embedding_client,
            convergence_threshold=threshold,
            thesaurus=thesaurus,
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
        "runtime": _runtime_details(spec.runtime_profile),
        "raw_response": json.dumps(recorder.records, indent=2, sort_keys=True),
    }


def _run_algo3(
    spec: HFRunSpec,
    *,
    hf_runtime: HFTransformersRuntimeFactory,
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
    recorder = _RecordingChatClient(chat_client)
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
            return _propose_children_from_prompt(
                recorder,
                _render_prompt(
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
        "runtime": _runtime_details(spec.runtime_profile),
        "raw_response": json.dumps(recorder.records, indent=2, sort_keys=True),
    }
