from __future__ import annotations

import subprocess
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm_conceptual_modeling.common.failure_markers import classify_failure
from llm_conceptual_modeling.common.io import coerce_int, read_json_dict, write_json_dict
from llm_conceptual_modeling.hf_batch.spec_path import SpecIdentity, run_dir_identity
from llm_conceptual_modeling.hf_state.active_models import resolve_active_chat_model_slugs
from llm_conceptual_modeling.hf_state.shard_manifest import manifest_identity_keys

_RETRYABLE_FAILURE_KINDS = {"timeout", "oom", "infrastructure", "structural"}


@dataclass
class _RunCounts:
    finished_count: int = 0
    failed_count: int = 0
    running_count: int = 0
    pending_count: int = 0
    failures: list[dict[str, object]] = field(default_factory=list)
    running_run_dirs: list[Path] = field(default_factory=list)


@dataclass(frozen=True)
class _ResolvedCounts:
    total_runs: int
    finished_count: int
    failed_count: int
    running_count: int
    pending_count: int


def collect_batch_status(output_root: str | Path) -> dict[str, object]:
    output_root_path = Path(output_root)
    status_file = _read_json(output_root_path / "batch_status.json")
    active_model_slugs = resolve_active_chat_model_slugs(output_root_path)
    manifest_identities = _load_manifest_identity_keys(output_root_path / "shard_manifest.json")
    planned_total_runs = _load_planned_total_runs(output_root_path)
    run_dirs = _matching_run_directories(
        output_root_path,
        active_model_slugs=active_model_slugs,
        manifest_identities=manifest_identities,
    )
    run_counts = _collect_run_counts(run_dirs)
    counts = _resolve_counts(
        status_file=status_file,
        run_counts=run_counts,
        run_count=len(run_dirs),
        planned_total_runs=planned_total_runs,
        manifest_identities=manifest_identities,
    )
    active_run = (
        _collect_active_run_details(run_counts.running_run_dirs[0])
        if run_counts.running_run_dirs
        else {}
    )
    return _status_payload(
        status_file=status_file,
        counts=counts,
        failures=run_counts.failures,
        active_run=active_run,
    )


def _matching_run_directories(
    output_root: Path,
    *,
    active_model_slugs: set[str],
    manifest_identities: set[SpecIdentity],
) -> list[Path]:
    return sorted(
        _iter_run_directories(
            output_root / "runs",
            active_model_slugs=active_model_slugs,
            manifest_identities=manifest_identities,
        )
    )


def _collect_run_counts(run_dirs: list[Path]) -> _RunCounts:
    counts = _RunCounts()
    for run_dir in run_dirs:
        _update_run_counts(counts, run_dir)
    return counts


def _update_run_counts(counts: _RunCounts, run_dir: Path) -> None:
    state = _read_json(run_dir / "state.json")
    raw_status = state.get("status")
    if raw_status is None:
        counts.pending_count += 1
        return
    status = str(raw_status)
    if status == "finished":
        counts.finished_count += 1
        return
    if status == "failed":
        _update_failed_run_counts(counts, run_dir)
        return
    if status == "running":
        counts.running_count += 1
        counts.running_run_dirs.append(run_dir)
        return
    counts.pending_count += 1


def _update_failed_run_counts(counts: _RunCounts, run_dir: Path) -> None:
    error = _read_json(run_dir / "error.json")
    failure_kind = classify_failure(
        error_type=str(error.get("type", "")),
        message=str(error.get("message", "")),
    )
    if failure_kind in _RETRYABLE_FAILURE_KINDS:
        counts.pending_count += 1
        return
    counts.failed_count += 1
    counts.failures.append(
        {
            "run_dir": str(run_dir),
            "message": error.get("message"),
            "type": error.get("type"),
        }
    )


def _resolve_counts(
    *,
    status_file: dict[str, Any],
    run_counts: _RunCounts,
    run_count: int,
    planned_total_runs: int,
    manifest_identities: set[SpecIdentity],
) -> _ResolvedCounts:
    if _has_explicit_batch_counts(status_file):
        finished_count = coerce_int(status_file.get("finished_count", run_counts.finished_count))
        failed_count = coerce_int(status_file.get("failed_count", run_counts.failed_count))
        running_count = coerce_int(status_file.get("running_count", run_counts.running_count))
        pending_count = coerce_int(status_file.get("pending_count", run_counts.pending_count))
        total_runs = coerce_int(status_file.get("total_runs", run_count))
    else:
        finished_count = run_counts.finished_count
        failed_count = run_counts.failed_count
        running_count = run_counts.running_count
        total_runs = _infer_total_runs(
            run_count=run_count,
            status_file=status_file,
            planned_total_runs=planned_total_runs,
            manifest_identities=manifest_identities,
        )
        inferred_pending_count = total_runs - finished_count - failed_count - running_count
        pending_count = max(inferred_pending_count, run_counts.pending_count)

    total_runs = _resolve_final_total_runs(
        total_runs=total_runs,
        status_file=status_file,
        planned_total_runs=planned_total_runs,
    )
    return _ResolvedCounts(
        total_runs=total_runs,
        finished_count=finished_count,
        failed_count=failed_count,
        running_count=running_count,
        pending_count=pending_count,
    )


def _has_explicit_batch_counts(status_file: dict[str, Any]) -> bool:
    return any(
        key in status_file
        for key in ("finished_count", "failed_count", "running_count", "pending_count")
    )


def _infer_total_runs(
    *,
    run_count: int,
    status_file: dict[str, Any],
    planned_total_runs: int,
    manifest_identities: set[SpecIdentity],
) -> int:
    total_runs = run_count
    if total_runs == 0:
        total_runs = coerce_int(status_file.get("total_runs"))
    if total_runs <= 0:
        total_runs = planned_total_runs
    if manifest_identities:
        total_runs = len(manifest_identities)
    return total_runs


def _resolve_final_total_runs(
    *,
    total_runs: int,
    status_file: dict[str, Any],
    planned_total_runs: int,
) -> int:
    if total_runs <= 0:
        total_runs = coerce_int(status_file.get("total_runs"))
    if total_runs <= 0:
        total_runs = planned_total_runs
    return total_runs


def _status_payload(
    *,
    status_file: dict[str, Any],
    counts: _ResolvedCounts,
    failures: list[dict[str, object]],
    active_run: dict[str, object],
) -> dict[str, object]:
    percent_complete = (
        round((counts.finished_count / counts.total_runs) * 100.0, 2)
        if counts.total_runs
        else 0.0
    )
    failure_type_counts = dict(
        Counter(str(failure.get("type")) for failure in failures if failure.get("type"))
    )
    return {
        "total_runs": counts.total_runs,
        "finished_count": counts.finished_count,
        "failed_count": counts.failed_count,
        "running_count": counts.running_count,
        "pending_count": counts.pending_count,
        "failure_count": len(failures),
        "failures": failures,
        "percent_complete": percent_complete,
        "current_run": status_file.get("current_run"),
        "last_completed_run": status_file.get("last_completed_run"),
        "started_at": status_file.get("started_at"),
        "updated_at": status_file.get("updated_at"),
        "active_stage": active_run.get("active_stage"),
        "active_stage_age_seconds": active_run.get("active_stage_age_seconds"),
        "worker_pid": active_run.get("worker_pid"),
        "worker_status": active_run.get("worker_status"),
        "worker_loaded_model": active_run.get("worker_loaded_model"),
        "gpu_processes": _query_gpu_processes(),
        "failure_types": failure_type_counts,
    }


def current_run_payload(
    *,
    algorithm: str,
    model: str,
    decoding_algorithm: str,
    graph_source: str = "default",
    pair_name: str,
    condition_bits: str,
    replication: int,
) -> dict[str, object]:
    return {
        "algorithm": algorithm,
        "model": model,
        "decoding_algorithm": decoding_algorithm,
        "graph_source": graph_source,
        "pair_name": pair_name,
        "condition_bits": condition_bits,
        "replication": replication,
    }


def _iter_run_directories(
    runs_root: Path,
    *,
    active_model_slugs: set[str],
    manifest_identities: set[SpecIdentity],
) -> list[Path]:
    if not runs_root.exists():
        return []
    return [
        path
        for path in runs_root.rglob("rep_*")
        if path.is_dir()
        and _run_dir_matches_filters(
            runs_root,
            path,
            active_model_slugs=active_model_slugs,
            manifest_identities=manifest_identities,
        )
    ]


def _read_json(path: Path) -> dict[str, Any]:
    return read_json_dict(path)


def _run_dir_matches_filters(
    runs_root: Path,
    run_dir: Path,
    active_model_slugs: set[str],
    *,
    manifest_identities: set[SpecIdentity],
) -> bool:
    parsed_identity = run_dir_identity(runs_root=runs_root, run_dir=run_dir)
    if parsed_identity is None:
        return False
    model_slug, identity = parsed_identity
    if active_model_slugs and model_slug not in active_model_slugs:
        return False
    if not manifest_identities:
        return True
    return identity in manifest_identities


def _load_manifest_identity_keys(manifest_path: Path) -> set[SpecIdentity]:
    return manifest_identity_keys(read_json_dict(manifest_path))


def _load_planned_total_runs(output_root: Path) -> int:
    preview_plan_path = output_root / "preview" / "resolved_run_plan.json"
    preview_plan = _read_json(preview_plan_path)
    return coerce_int(preview_plan.get("planned_total_runs"))


def _collect_active_run_details(run_dir: Path) -> dict[str, object]:
    active_stage = _read_json(run_dir / "active_stage.json")
    worker_state = _read_json(run_dir / "worker_state.json")
    stage_age_seconds = _active_stage_age_seconds(run_dir, worker_state)
    return {
        "active_stage": active_stage or None,
        "active_stage_age_seconds": stage_age_seconds,
        "worker_pid": worker_state.get("worker_pid") or worker_state.get("pid"),
        "worker_status": worker_state.get("status"),
        "worker_loaded_model": worker_state.get("model_loaded") is True,
    }


def _active_stage_age_seconds(
    run_dir: Path,
    worker_state: dict[str, Any],
) -> float | None:
    active_stage_path = run_dir / "active_stage.json"
    if active_stage_path.exists():
        return round(time.time() - active_stage_path.stat().st_mtime, 3)
    worker_state_path = run_dir / "worker_state.json"
    if worker_state_path.exists() and worker_state.get("status") == "running":
        return round(time.time() - worker_state_path.stat().st_mtime, 3)
    return None


def _query_gpu_processes() -> list[dict[str, object]]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return _parse_gpu_process_output(completed.stdout)


def _parse_gpu_process_output(output: str) -> list[dict[str, object]]:
    processes: list[dict[str, object]] = []
    for line in output.splitlines():
        process = _parse_gpu_process_line(line)
        if process is not None:
            processes.append(process)
    return processes


def _parse_gpu_process_line(line: str) -> dict[str, object] | None:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 2 or not parts[0]:
        return None
    pid = _parse_gpu_pid(parts[0])
    if pid is None:
        return None
    return {
        "pid": pid,
        "used_gpu_memory_mib": _parse_gpu_memory(parts[1]),
    }


def _parse_gpu_pid(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _parse_gpu_memory(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def status_timestamp_now() -> str:
    return datetime.now(UTC).isoformat()


def write_status_snapshot(*, output_root: str | Path, status: dict[str, object]) -> None:
    output_root_path = Path(output_root)
    status_path = output_root_path / "batch_status.json"
    write_json_dict(status_path, status)
