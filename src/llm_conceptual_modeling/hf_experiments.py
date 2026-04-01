from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, cast

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
from llm_conceptual_modeling.common.connection_eval import find_valid_connections
from llm_conceptual_modeling.common.graph_data import load_algo2_thesaurus
from llm_conceptual_modeling.common.hf_transformers import (
    DecodingConfig,
    HFTransformersRuntimeFactory,
    RuntimeProfile,
    build_runtime_factory,
)
from llm_conceptual_modeling.common.literals import parse_python_literal
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
from llm_conceptual_modeling.hf_batch.run_artifacts import (
    clear_retry_artifacts as _clear_retry_artifacts,
)
from llm_conceptual_modeling.hf_batch.run_artifacts import (
    normalize_stale_running_run as _normalize_stale_running_run,
)
from llm_conceptual_modeling.hf_batch.run_artifacts import (
    read_json as _read_artifact_json,
)
from llm_conceptual_modeling.hf_batch.run_artifacts import (
    write_smoke_verdict as _write_smoke_verdict,
)
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
from llm_conceptual_modeling.hf_spec_codec import serialize_spec
from llm_conceptual_modeling.hf_subprocess import (
    build_hf_download_environment,
    run_monitored_command,
)


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

    use_monitored_hf_subprocess = runtime_factory is None and not dry_run
    hf_runtime = None
    if runtime_factory is None and not use_monitored_hf_subprocess:
        hf_runtime = build_runtime_factory(hf_token=_resolve_hf_token())
    if dry_run or use_monitored_hf_subprocess:
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
    if runtime_factory is None and not use_monitored_hf_subprocess:
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

        if resume:
            _normalize_stale_running_run(run_dir)
        cached = _load_valid_finished_summary(run_dir=run_dir, algorithm=spec.algorithm)
        if resume and cached is not None:
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

        _clear_retry_artifacts(run_dir)
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
            if use_monitored_hf_subprocess:
                runtime_result = _run_local_hf_spec_subprocess(spec=spec, run_dir=run_dir)
            else:
                if runtime_factory is None:
                    raise ValueError("Missing runtime_factory for in-process execution.")
                runtime_result = _execute_run(
                    spec=spec,
                    runtime_factory=runtime_factory,
                    dry_run=dry_run,
                    run_dir=run_dir,
                )
            if not dry_run:
                _validate_structural_runtime_result(
                    algorithm=spec.algorithm,
                    raw_row=runtime_result["raw_row"],
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
            continue

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
            **_summary_from_raw_row(spec.algorithm, raw_row),
            **runtime_result.get("summary", {}),
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
    if summary_frame.empty:
        return

    write_aggregated_outputs(output_root_path, summary_frame)


def select_run_spec(
    *,
    config: HFRunConfig,
    algorithm: str,
    model: str,
    pair_name: str,
    condition_bits: str,
    decoding: DecodingConfig,
    replication: int,
    runtime_profile_provider: Callable[[str], RuntimeProfile] | None = None,
) -> HFRunSpec:
    specs = plan_paper_batch(
        models=[model],
        embedding_model=config.models.embedding_model,
        replications=max(replication + 1, 1),
        algorithms=(algorithm,),
        config=config,
        runtime_profile_provider=runtime_profile_provider,
    )
    for spec in specs:
        if (
            spec.algorithm == algorithm
            and spec.model == model
            and spec.pair_name == pair_name
            and spec.condition_bits == condition_bits
            and spec.replication == replication
            and spec.decoding == decoding
        ):
            return spec
    raise ValueError(
        "No configured run spec matches "
        f"algorithm={algorithm!r}, model={model!r}, pair_name={pair_name!r}, "
        f"condition_bits={condition_bits!r}, decoding={decoding!r}, "
        f"replication={replication!r}."
    )


def run_single_spec(
    *,
    spec: HFRunSpec,
    output_root: str | Path,
    runtime_factory: RuntimeFactory | None = None,
    dry_run: bool = False,
    resume: bool = False,
) -> dict[str, object]:
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    use_monitored_hf_subprocess = runtime_factory is None and not dry_run

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

    if resume:
        _normalize_stale_running_run(run_dir)
    cached_summary = _load_valid_finished_summary(run_dir=run_dir, algorithm=spec.algorithm)
    if resume and cached_summary is not None:
        _write_smoke_verdict(
            output_root=output_root_path,
            run_dir=run_dir,
            spec_identity=_smoke_spec_identity(spec),
            status="success",
            worker_loaded_model=True,
        )
        return cached_summary

    _clear_retry_artifacts(run_dir)
    _write_json(run_dir / "manifest.json", _manifest_for_spec(spec))
    _write_json(run_dir / "state.json", {"status": "running"})
    try:
        if use_monitored_hf_subprocess:
            runtime_result = _run_local_hf_spec_subprocess(spec=spec, run_dir=run_dir)
        else:
            if runtime_factory is None:
                hf_runtime = build_runtime_factory(hf_token=_resolve_hf_token())
                runtime_factory = _runtime_factory_from_hf_runtime(hf_runtime)
            runtime_result = _execute_run(
                spec=spec,
                runtime_factory=runtime_factory,
                dry_run=dry_run,
                run_dir=run_dir,
            )
        if not dry_run:
            _validate_structural_runtime_result(
                algorithm=spec.algorithm,
                raw_row=runtime_result["raw_row"],
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
        _write_smoke_verdict(
            output_root=output_root_path,
            run_dir=run_dir,
            spec_identity=_smoke_spec_identity(spec),
            status="failed",
            failure_type=type(error).__name__,
            failure_message=str(error),
            worker_loaded_model=_worker_loaded_model(run_dir),
        )
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
        **_summary_from_raw_row(spec.algorithm, raw_row),
        **runtime_result.get("summary", {}),
    }
    _write_json(summary_path, summary)
    _write_smoke_verdict(
        output_root=output_root_path,
        run_dir=run_dir,
        spec_identity=_smoke_spec_identity(spec),
        status="success",
        worker_loaded_model=_worker_loaded_model(run_dir) or True,
    )
    return summary


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


def _load_valid_finished_summary(
    *,
    run_dir: Path,
    algorithm: str,
) -> dict[str, object] | None:
    if not _is_finished_run_directory(run_dir):
        return None

    summary_path = run_dir / "summary.json"
    raw_row_path = run_dir / "raw_row.json"
    try:
        raw_row = json.loads(raw_row_path.read_text(encoding="utf-8"))
        _validate_structural_runtime_result(algorithm=algorithm, raw_row=raw_row)
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
        if summary_path.exists():
            summary_path.unlink()
        return None

    return json.loads(summary_path.read_text(encoding="utf-8"))


def _smoke_spec_identity(spec: HFRunSpec) -> dict[str, object]:
    return {
        "algorithm": spec.algorithm,
        "model": spec.model,
        "embedding_model": spec.embedding_model,
        "decoding_algorithm": spec.decoding.algorithm,
        "pair_name": spec.pair_name,
        "condition_bits": spec.condition_bits,
        "replication": spec.replication,
    }


def _worker_loaded_model(run_dir: Path) -> bool:
    worker_state = _read_artifact_json(run_dir / "worker_state.json")
    return bool(worker_state.get("model_loaded")) or (run_dir / "active_stage.json").exists()


def _execute_run(
    *,
    spec: HFRunSpec,
    runtime_factory: RuntimeFactory,
    dry_run: bool,
    run_dir: Path | None = None,
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
    try:
        runtime_callable = cast(Any, runtime_factory)
        return runtime_callable(spec, run_dir=run_dir)
    except TypeError as error:
        if "run_dir" not in str(error):
            raise
        return runtime_factory(spec)


def _run_local_hf_spec_subprocess(*, spec: HFRunSpec, run_dir: Path) -> RuntimeResult:
    startup_timeout_seconds = _resolve_startup_timeout_seconds(spec.context_policy)
    stage_timeout_seconds = _resolve_stage_timeout_seconds(spec.context_policy)
    retry_attempts = _resolve_run_retry_attempts(spec.context_policy)
    spec_json_path = run_dir / "worker_spec.json"
    result_json_path = run_dir / "worker_result.json"

    for attempt in range(1, retry_attempts + 1):
        _clear_retry_artifacts(run_dir)
        _write_json(spec_json_path, serialize_spec(spec))
        completed = run_monitored_command(
            command=_build_worker_command(
                spec_json_path=spec_json_path,
                result_json_path=result_json_path,
                run_dir=run_dir,
            ),
            run_dir=run_dir,
            startup_timeout_seconds=startup_timeout_seconds,
            stage_timeout_seconds=stage_timeout_seconds,
            env=build_hf_download_environment(),
        )
        if not result_json_path.exists():
            raise RuntimeError(
                "HF worker subprocess exited without writing a result artifact. "
                f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
            )
        worker_payload = json.loads(result_json_path.read_text(encoding="utf-8"))
        if worker_payload.get("ok"):
            return cast(RuntimeResult, worker_payload["runtime_result"])

        error = cast(dict[str, str], worker_payload["error"])
        if attempt < retry_attempts and _is_retryable_worker_error(error):
            continue
        raise RuntimeError(f"{error['type']}: {error['message']}")

    raise RuntimeError("HF worker retry loop exhausted without a terminal result.")


def _build_worker_command(
    *,
    spec_json_path: Path,
    result_json_path: Path,
    run_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "llm_conceptual_modeling.hf_worker",
        "--spec-json",
        str(spec_json_path),
        "--result-json",
        str(result_json_path),
        "--run-dir",
        str(run_dir),
    ]


def _resolve_startup_timeout_seconds(context_policy: dict[str, object] | None) -> float:
    if context_policy is None:
        return 900.0
    raw_value = context_policy.get("startup_timeout_seconds", 900)
    return _coerce_timeout_seconds(raw_value)


def _resolve_stage_timeout_seconds(context_policy: dict[str, object] | None) -> float:
    if context_policy is None:
        return 180.0
    raw_value = context_policy.get("generation_timeout_seconds", 180)
    return _coerce_timeout_seconds(raw_value)


def _resolve_run_retry_attempts(context_policy: dict[str, object] | None) -> int:
    if context_policy is None:
        return 2
    raw_value = context_policy.get("run_retry_attempts", 2)
    if isinstance(raw_value, bool):
        raise TypeError("Retry attempts value must be numeric, got bool")
    if isinstance(raw_value, (int, float, str)):
        attempts = int(float(raw_value))
        return max(1, attempts)
    raise TypeError(f"Retry attempts value must be numeric, got {type(raw_value).__name__}")


def _is_retryable_worker_error(error: dict[str, str]) -> bool:
    if error.get("type") != "ValueError":
        return False
    message = error.get("message", "")
    retry_markers = (
        "Model did not return valid structured output:",
        "Invalid edge item shape:",
        "Unsupported structured response shape for schema",
        "Structured edge_list response must contain a list of edges",
        "Structured edge_list flat string response must contain an even number of items",
        "Structurally invalid algo",
    )
    return any(marker in message for marker in retry_markers)


def _coerce_timeout_seconds(raw_value: object) -> float:
    if isinstance(raw_value, (int, float, str)):
        return float(raw_value)
    raise TypeError(f"Timeout value must be numeric, got {type(raw_value).__name__}")


def _runtime_factory_from_hf_runtime(hf_runtime: HFTransformersRuntimeFactory) -> RuntimeFactory:
    def runtime(spec: HFRunSpec, *, run_dir: Path | None = None) -> RuntimeResult:
        if spec.algorithm == "algo1":
            return _run_algo1(spec, hf_runtime=hf_runtime, run_dir=run_dir)
        if spec.algorithm == "algo2":
            return _run_algo2(spec, hf_runtime=hf_runtime, run_dir=run_dir)
        if spec.algorithm == "algo3":
            return _run_algo3(spec, hf_runtime=hf_runtime, run_dir=run_dir)
        raise ValueError(f"Unsupported algorithm: {spec.algorithm}")

    return runtime


def _run_algo1(
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
    recorder = _RecordingChatClient(
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
            candidate_edges = _generate_edges_from_prompt(
                recorder,
                _render_prompt(
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
        verified_edges = _verify_edges_from_prompt(
            recorder,
            _render_prompt(
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
    _validate_algorithm_edge_result("algo1", result.verified_edges)
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
        "summary": {
            "candidate_edge_count": len(result.candidate_edges),
            "verified_edge_count": len(result.verified_edges),
            **_connection_metric_summary(raw_row),
        },
    }


def _run_algo2(
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
    recorder = _RecordingChatClient(
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
        "runtime": _runtime_details(spec.runtime_profile),
        "raw_response": json.dumps(recorder.records, indent=2, sort_keys=True),
        "summary": {
            "candidate_edge_count": len(result.raw_edges),
            "verified_edge_count": len(result.normalized_edges),
            **_connection_metric_summary(raw_row),
        },
    }


def _run_algo3(
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
    recorder = _RecordingChatClient(
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
        "summary": {
            "result_edge_count": len(result_edges),
            "recall": float(raw_row["Recall"]),
        },
    }


def _connection_metric_summary(raw_row: dict[str, object]) -> dict[str, float]:
    graph = parse_python_literal(str(raw_row["graph"]))
    subgraph1 = parse_python_literal(str(raw_row["subgraph1"]))
    subgraph2 = parse_python_literal(str(raw_row["subgraph2"]))
    result_edges = parse_python_literal(str(raw_row["Result"]))
    ground_truth_connections = find_valid_connections(graph, subgraph1, subgraph2)
    proposed_edges = list(subgraph1) + list(subgraph2) + list(result_edges)
    generated_connections = find_valid_connections(proposed_edges, subgraph1, subgraph2)
    nodes1 = {node for edge in subgraph1 for node in edge}
    nodes2 = {node for edge in subgraph2 for node in edge}
    tp = len(generated_connections & ground_truth_connections)
    fp = len(generated_connections - ground_truth_connections)
    fn = len(ground_truth_connections - generated_connections)
    tn = (len(nodes1) * len(nodes2)) - (tp + fp + fn)
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    denominator = precision + recall
    f1 = (2 * precision * recall) / denominator if denominator > 0 else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _summary_from_raw_row(algorithm: str, raw_row: dict[str, object]) -> dict[str, object]:
    if algorithm in {"algo1", "algo2"} and "Result" in raw_row:
        result_edges = parse_python_literal(str(raw_row["Result"]))
        result_count = len(result_edges)
        return {
            "candidate_edge_count": result_count,
            "verified_edge_count": result_count,
            **_connection_metric_summary(raw_row),
        }
    if algorithm == "algo3" and "Results" in raw_row:
        result_edges = parse_python_literal(str(raw_row["Results"]))
        return {
            "result_edge_count": len(result_edges),
            "recall": float(str(raw_row.get("Recall", 0.0))),
        }
    return {}


def _validate_structural_runtime_result(*, algorithm: str, raw_row: dict[str, object]) -> None:
    if algorithm != "algo1":
        return
    result_literal = raw_row.get("Result")
    if result_literal is None:
        raise ValueError(f"Structurally invalid {algorithm} result: missing Result field.")
    result_edges = parse_python_literal(str(result_literal))
    _validate_algorithm_edge_result(algorithm, result_edges)


def _validate_algorithm_edge_result(algorithm: str, result_edges: object) -> None:
    if not isinstance(result_edges, list):
        raise ValueError(f"Structurally invalid {algorithm} result: Result is not a list.")
    if not result_edges:
        raise ValueError(
            f"Structurally invalid {algorithm} result: verified edge list is empty."
        )
    for edge in result_edges:
        if not isinstance(edge, (list, tuple)) or len(edge) < 2:
            raise ValueError(
                f"Structurally invalid {algorithm} result: invalid edge shape {edge!r}."
            )
        source = _normalize_structural_endpoint(edge[0], algorithm=algorithm)
        target = _normalize_structural_endpoint(edge[1], algorithm=algorithm)
        if not _looks_like_textual_concept(source) or not _looks_like_textual_concept(target):
            raise ValueError(
                "Structurally invalid "
                f"{algorithm} result: non-textual edge endpoint {edge!r}."
            )


def _normalize_structural_endpoint(value: object, *, algorithm: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Structurally invalid {algorithm} result: empty edge endpoint.")
    return text


def _looks_like_textual_concept(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text))
