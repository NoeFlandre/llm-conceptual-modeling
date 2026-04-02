from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, cast

from llm_conceptual_modeling.hf_batch_types import HFRunSpec

JsonObject = dict[str, object]


def _noop_normalize_stale_running_run(run_dir: Path) -> None:
    _ = run_dir


def build_seeded_resume_snapshot(
    *,
    output_root: Path,
    planned_specs: list[HFRunSpec],
    started_at: str,
    run_dir_for_spec_fn: Callable[[Path, HFRunSpec], Path],
    current_run_payload_fn: Callable[..., JsonObject],
    status_timestamp_now_fn: Callable[[], str],
    validate_structural_runtime_result_fn: Callable[..., None],
    normalize_stale_running_run_fn: Callable[[Path], object | None] | None = None,
    read_artifact_json_fn: Callable[[Path], JsonObject] | None = None,
    write_json_fn: Callable[[Path, dict[str, Any]], None] | None = None,
) -> tuple[JsonObject, list[JsonObject], set[Path], set[Path]]:
    summary_rows: list[JsonObject] = []
    seeded_finished_run_dirs: set[Path] = set()
    seeded_failed_run_dirs: set[Path] = set()
    failures: list[JsonObject] = []
    last_completed_run: JsonObject | None = None

    if read_artifact_json_fn is None:
        read_artifact_json_fn = read_artifact_json
    if write_json_fn is None:
        write_json_fn = write_json
    if normalize_stale_running_run_fn is None:
        normalize_stale_running_run_fn = _noop_normalize_stale_running_run

    previous_status = read_artifact_json_fn(output_root / "batch_status.json")

    for spec in planned_specs:
        run_dir = run_dir_for_spec_fn(output_root, spec)
        normalize_stale_running_run_fn(run_dir)

        cached = load_valid_finished_summary(
            run_dir=run_dir,
            algorithm=spec.algorithm,
            validate_structural_runtime_result_fn=validate_structural_runtime_result_fn,
            write_json_fn=write_json_fn,
        )
        if cached is not None:
            summary_rows.append(cached)
            seeded_finished_run_dirs.add(run_dir)
            last_completed_run = current_run_payload_fn(
                algorithm=spec.algorithm,
                model=spec.model,
                decoding_algorithm=spec.decoding.algorithm,
                pair_name=spec.pair_name,
                condition_bits=spec.condition_bits,
                replication=spec.replication,
            )
            continue

        deferred_failure = load_deferred_failed_summary(
            run_dir=run_dir,
            algorithm=spec.algorithm,
            context_policy=spec.context_policy,
            read_artifact_json_fn=read_artifact_json_fn,
        )
        if deferred_failure is not None:
            failures.append(deferred_failure)
            seeded_failed_run_dirs.add(run_dir)

    finished_count = len(seeded_finished_run_dirs)
    failed_count = len(seeded_failed_run_dirs)
    total_runs = len(planned_specs)
    pending_count = max(total_runs - finished_count - failed_count, 0)
    percent_complete = round((finished_count / total_runs) * 100.0, 2) if total_runs else 0.0

    status_snapshot: JsonObject = {
        "total_runs": total_runs,
        "finished_count": finished_count,
        "failed_count": failed_count,
        "running_count": 0,
        "pending_count": pending_count,
        "failure_count": len(failures),
        "failures": failures,
        "percent_complete": percent_complete,
        "current_run": None,
        "last_completed_run": last_completed_run or previous_status.get("last_completed_run"),
        "started_at": previous_status.get("started_at", started_at),
        "updated_at": status_timestamp_now_fn(),
    }
    return status_snapshot, summary_rows, seeded_finished_run_dirs, seeded_failed_run_dirs


def status_int(status_snapshot: JsonObject, key: str) -> int:
    return int(cast(int | float | str, status_snapshot[key]))


def status_failures(status_snapshot: JsonObject) -> list[JsonObject]:
    return list(cast(list[JsonObject], status_snapshot["failures"]))


def is_finished_run_directory(run_dir: Path) -> bool:
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


def load_valid_finished_summary(
    *,
    run_dir: Path,
    algorithm: str,
    validate_structural_runtime_result_fn: Callable[..., None],
    write_json_fn: Callable[[Path, dict[str, Any]], None] | None = None,
) -> JsonObject | None:
    if not is_finished_run_directory(run_dir):
        return None

    if write_json_fn is None:
        write_json_fn = write_json

    summary_path = run_dir / "summary.json"
    raw_row_path = run_dir / "raw_row.json"
    try:
        raw_row = json.loads(raw_row_path.read_text(encoding="utf-8"))
        validate_structural_runtime_result_fn(algorithm=algorithm, raw_row=raw_row)
    except Exception as error:
        write_json_fn(
            run_dir / "error.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "status": "failed",
            },
        )
        write_json_fn(run_dir / "state.json", {"status": "failed"})
        if summary_path.exists():
            summary_path.unlink()
        return None

    return cast(JsonObject, json.loads(summary_path.read_text(encoding="utf-8")))


def load_deferred_failed_summary(
    *,
    run_dir: Path,
    algorithm: str,
    context_policy: dict[str, object] | None,
    read_artifact_json_fn: Callable[[Path], JsonObject] | None = None,
) -> JsonObject | None:
    _ = algorithm
    if read_artifact_json_fn is None:
        read_artifact_json_fn = read_artifact_json
    state = read_artifact_json_fn(run_dir / "state.json")
    if state.get("status") != "failed":
        return None
    error = read_artifact_json_fn(run_dir / "error.json")
    failure_kind = classify_failure_payload(error)
    if failure_kind == "timeout" and not resolve_retry_timeout_failures_on_resume(context_policy):
        pass
    elif failure_kind == "oom" and not resolve_retry_oom_failures_on_resume(context_policy):
        pass
    elif failure_kind == "unsupported":
        pass
    else:
        return None
    return {
        "run_dir": str(run_dir),
        "message": str(error.get("message", "Deferred failure during resume.")),
        "type": str(error.get("type", "RuntimeError")),
        "failure_kind": failure_kind,
        "deferred_on_resume": True,
    }


def resolve_retry_timeout_failures_on_resume(
    context_policy: dict[str, object] | None,
) -> bool:
    mode = resolve_resume_pass_mode(context_policy)
    if mode == "retry-timeouts":
        return True
    if context_policy is None:
        return False
    raw_value = context_policy.get("retry_timeout_failures_on_resume", False)
    if isinstance(raw_value, bool):
        return raw_value
    raise TypeError(
        "retry_timeout_failures_on_resume value must be boolean, "
        f"got {type(raw_value).__name__}"
    )


def resolve_retry_oom_failures_on_resume(
    context_policy: dict[str, object] | None,
) -> bool:
    if context_policy is None:
        return True
    raw_value = context_policy.get("retry_oom_failures_on_resume", True)
    if isinstance(raw_value, bool):
        return raw_value
    raise TypeError(
        "retry_oom_failures_on_resume value must be boolean, "
        f"got {type(raw_value).__name__}"
    )


def resolve_resume_pass_mode(context_policy: dict[str, object] | None) -> str:
    if context_policy is None:
        return "throughput"
    raw_value = context_policy.get("resume_pass_mode", "throughput")
    if not isinstance(raw_value, str):
        raise TypeError(
            "resume_pass_mode value must be string, "
            f"got {type(raw_value).__name__}"
        )
    normalized = raw_value.strip().lower()
    if normalized not in {"throughput", "retry-timeouts"}:
        raise ValueError(
            "resume_pass_mode must be one of {'throughput', 'retry-timeouts'}, "
            f"got {raw_value!r}"
        )
    return normalized


def classify_failure_payload(error: JsonObject) -> str:
    error_type = str(error.get("type", ""))
    message = str(error.get("message", ""))
    if error_type == "MonitoredCommandTimeout" or "MonitoredCommandTimeout" in message:
        return "timeout"
    lowered_message = message.lower()
    if "out of memory" in lowered_message:
        return "oom"
    if (
        "contrastive search is not supported with stateful models" in lowered_message
        or "contrastive search requires `trust_remote_code=true`" in lowered_message
    ):
        return "unsupported"
    if error_type == "StaleRunState":
        return "stale"
    if error_type in {"ValueError", "RuntimeError"} and (
        "Model did not return valid structured output:" in message
        or "Could not parse tuple content:" in message
        or "Invalid edge item shape:" in message
        or "Unsupported structured response shape for schema" in message
        or "Structured edge_list response must contain a list of edges" in message
        or "Structured edge_list flat string response must contain an even number of items"
        in message
        or "Structurally invalid algo" in message
    ):
        return "structural"
    return "other"


def order_planned_specs_for_resume(
    *,
    planned_specs: list[HFRunSpec],
    output_root: Path,
    resume: bool,
    run_dir_for_spec_fn: Callable[[Path, HFRunSpec], Path],
    read_artifact_json_fn: Callable[[Path], JsonObject] | None = None,
) -> list[HFRunSpec]:
    if not resume or not planned_specs:
        return planned_specs
    history = collect_resume_history(
        output_root=output_root,
        read_artifact_json_fn=read_artifact_json_fn,
    )
    return sorted(
        planned_specs,
        key=lambda spec: resume_priority_key(
            spec=spec,
            run_dir=run_dir_for_spec_fn(output_root, spec),
            history=history,
            read_artifact_json_fn=read_artifact_json_fn,
        ),
    )


def collect_resume_history(
    output_root: Path,
    read_artifact_json_fn: Callable[[Path], JsonObject] | None = None,
) -> dict[str, dict[object, dict[str, int]]]:
    if read_artifact_json_fn is None:
        read_artifact_json_fn = read_artifact_json
    by_pair: dict[object, dict[str, int]] = defaultdict(
        lambda: {"finished": 0, "timeout": 0, "structural": 0, "unsupported": 0, "other": 0}
    )
    by_pair_condition: dict[object, dict[str, int]] = defaultdict(
        lambda: {"finished": 0, "timeout": 0, "structural": 0, "unsupported": 0, "other": 0}
    )
    by_family: dict[object, dict[str, int]] = defaultdict(
        lambda: {"finished": 0, "timeout": 0, "structural": 0, "unsupported": 0, "other": 0}
    )
    runs_root = output_root / "runs"
    if not runs_root.exists():
        return {
            "by_pair": dict(by_pair),
            "by_pair_condition": dict(by_pair_condition),
            "by_family": dict(by_family),
        }
    for run_dir in runs_root.glob("*/*/*/*/*/rep_*"):
        condition_label = run_dir.parent.parent.parent.name
        pair_name = run_dir.parent.parent.name
        condition_bits = run_dir.parent.name
        pair_key = pair_name
        pair_condition_key = (pair_name, condition_bits)
        family_key = (pair_name, condition_bits, condition_label)
        state = read_artifact_json_fn(run_dir / "state.json")
        if state.get("status") == "finished":
            by_pair[pair_key]["finished"] += 1
            by_pair_condition[pair_condition_key]["finished"] += 1
            by_family[family_key]["finished"] += 1
            continue
        if state.get("status") != "failed":
            continue
        failure_kind = classify_failure_payload(read_artifact_json_fn(run_dir / "error.json"))
        bucket = (
            failure_kind
            if failure_kind in {"timeout", "structural", "unsupported"}
            else "other"
        )
        by_pair[pair_key][bucket] += 1
        by_pair_condition[pair_condition_key][bucket] += 1
        by_family[family_key][bucket] += 1
    return {
        "by_pair": dict(by_pair),
        "by_pair_condition": dict(by_pair_condition),
        "by_family": dict(by_family),
    }


def resume_priority_key(
    *,
    spec: HFRunSpec,
    run_dir: Path,
    history: dict[str, dict[object, dict[str, int]]],
    read_artifact_json_fn: Callable[[Path], JsonObject] | None = None,
) -> tuple[float, float, float, float, str, str, int]:
    if read_artifact_json_fn is None:
        read_artifact_json_fn = read_artifact_json
    context_policy = spec.context_policy
    pass_mode = resolve_resume_pass_mode(context_policy)
    state = read_artifact_json_fn(run_dir / "state.json")
    failure_kind = "pending"
    if state.get("status") == "failed":
        failure_kind = classify_failure_payload(read_artifact_json_fn(run_dir / "error.json"))

    if pass_mode == "retry-timeouts":
        base_bucket_map = {
            "timeout": 0.0,
            "structural": 1.0,
            "pending": 2.0,
            "other": 3.0,
            "unsupported": 4.0,
        }
    else:
        base_bucket_map = {
            "pending": 0.0,
            "structural": 1.0,
            "other": 2.0,
            "timeout": 3.0,
            "unsupported": 4.0,
        }

    pair_stats = history["by_pair"].get(spec.pair_name, {})
    condition_stats = history["by_pair_condition"].get((spec.pair_name, spec.condition_bits), {})
    family_stats = history["by_family"].get(
        (spec.pair_name, spec.condition_bits, spec.condition_label),
        {},
    )

    pair_finished = int(pair_stats.get("finished", 0))
    pair_failures = (
        int(pair_stats.get("timeout", 0))
        + int(pair_stats.get("unsupported", 0))
        + int(pair_stats.get("other", 0))
    )
    pair_total = pair_finished + pair_failures + int(pair_stats.get("structural", 0))
    pair_timeout_rate = int(pair_stats.get("timeout", 0)) / pair_total if pair_total else 0.0

    condition_finished = int(condition_stats.get("finished", 0))
    condition_failures = (
        int(condition_stats.get("timeout", 0))
        + int(condition_stats.get("structural", 0))
        + int(condition_stats.get("unsupported", 0))
        + int(condition_stats.get("other", 0))
    )
    condition_total = condition_finished + condition_failures
    condition_failure_rate = condition_failures / condition_total if condition_total else 0.0

    family_finished = int(family_stats.get("finished", 0))
    family_failures = (
        int(family_stats.get("timeout", 0))
        + int(family_stats.get("structural", 0))
        + int(family_stats.get("unsupported", 0))
        + int(family_stats.get("other", 0))
    )
    family_total = family_finished + family_failures
    family_failure_rate = family_failures / family_total if family_total else 0.0

    return (
        base_bucket_map.get(failure_kind, 4.0),
        family_failure_rate,
        pair_timeout_rate,
        condition_failure_rate,
        spec.pair_name,
        spec.condition_bits,
        spec.replication,
    )


def read_artifact_json(path: Path) -> JsonObject:
    if not path.exists():
        return {}
    return cast(JsonObject, json.loads(path.read_text(encoding="utf-8")))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
