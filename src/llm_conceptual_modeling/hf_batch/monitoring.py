from __future__ import annotations

import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm_conceptual_modeling.common.failure_markers import classify_failure
from llm_conceptual_modeling.common.io import coerce_int, read_json_dict, write_json_dict
from llm_conceptual_modeling.hf_state.active_models import resolve_active_chat_model_slugs
from llm_conceptual_modeling.hf_state.shard_manifest import manifest_identity_keys

_RETRYABLE_FAILURE_KINDS = {"timeout", "oom", "infrastructure", "structural"}


def collect_batch_status(output_root: str | Path) -> dict[str, object]:
    output_root_path = Path(output_root)
    runs_root = output_root_path / "runs"
    active_model_slugs = resolve_active_chat_model_slugs(output_root_path)
    manifest_identities = _load_manifest_identity_keys(output_root_path / "shard_manifest.json")
    run_dirs = sorted(
        _iter_run_directories(
            runs_root,
            active_model_slugs=active_model_slugs,
            manifest_identities=manifest_identities,
        )
    )

    finished_count = 0
    failed_count = 0
    running_count = 0
    pending_count = 0
    failures: list[dict[str, object]] = []
    running_run_dirs: list[Path] = []

    for run_dir in run_dirs:
        state = _read_json(run_dir / "state.json")
        status = str(state.get("status", "pending"))
        if status == "finished":
            finished_count += 1
            continue
        if status == "failed":
            error = _read_json(run_dir / "error.json")
            failure_kind = classify_failure(
                error_type=str(error.get("type", "")),
                message=str(error.get("message", "")),
            )
            if failure_kind in _RETRYABLE_FAILURE_KINDS:
                pending_count += 1
                continue
            failed_count += 1
            failures.append(
                {
                    "run_dir": str(run_dir),
                    "message": error.get("message"),
                    "type": error.get("type"),
                }
            )
            continue
        if status == "running":
            running_count += 1
            running_run_dirs.append(run_dir)
            continue
        pending_count += 1

    total_runs = len(run_dirs)
    if total_runs == 0:
        status_file = _read_json(output_root_path / "batch_status.json")
        total_runs = coerce_int(status_file.get("total_runs", 0))
    if manifest_identities:
        total_runs = len(manifest_identities)

    status_file = _read_json(output_root_path / "batch_status.json")
    pending_count = max(total_runs - finished_count - failed_count - running_count, pending_count)
    percent_complete = round((finished_count / total_runs) * 100.0, 2) if total_runs else 0.0
    active_run = _collect_active_run_details(running_run_dirs[0]) if running_run_dirs else {}
    failure_type_counts = dict(
        Counter(str(failure.get("type")) for failure in failures if failure.get("type"))
    )

    return {
        "total_runs": total_runs,
        "finished_count": finished_count,
        "failed_count": failed_count,
        "running_count": running_count,
        "pending_count": pending_count,
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
    pair_name: str,
    condition_bits: str,
    replication: int,
) -> dict[str, object]:
    return {
        "algorithm": algorithm,
        "model": model,
        "decoding_algorithm": decoding_algorithm,
        "pair_name": pair_name,
        "condition_bits": condition_bits,
        "replication": replication,
    }


def _iter_run_directories(
    runs_root: Path,
    *,
    active_model_slugs: set[str],
    manifest_identities: set[tuple[str, str, str, str, str, int]],
) -> list[Path]:
    if not runs_root.exists():
        return []
    return [
        path
        for path in runs_root.rglob("rep_*")
        if path.is_dir()
        and _run_dir_matches_active_models(runs_root, path, active_model_slugs)
        and _run_dir_matches_manifest(runs_root, path, manifest_identities)
    ]


def _read_json(path: Path) -> dict[str, Any]:
    return read_json_dict(path)


def _run_dir_matches_active_models(
    runs_root: Path,
    run_dir: Path,
    active_model_slugs: set[str],
) -> bool:
    if not active_model_slugs:
        return True
    try:
        model_slug = run_dir.resolve().relative_to(runs_root.resolve()).parts[1]
    except Exception:
        return True
    return model_slug in active_model_slugs


def _run_dir_matches_manifest(
    runs_root: Path,
    run_dir: Path,
    manifest_identities: set[tuple[str, str, str, str, str, int]],
) -> bool:
    if not manifest_identities:
        return True
    try:
        parts = run_dir.resolve().relative_to(runs_root.resolve()).parts
        identity = (
            parts[0],
            parts[1].replace("__", "/"),
            parts[2],
            parts[3],
            parts[4],
            int(parts[5].removeprefix("rep_")),
        )
    except Exception:
        return False
    return identity in manifest_identities


def _load_manifest_identity_keys(manifest_path: Path) -> set[tuple[str, str, str, str, str, int]]:
    return manifest_identity_keys(read_json_dict(manifest_path))


def _collect_active_run_details(run_dir: Path) -> dict[str, object]:
    active_stage = _read_json(run_dir / "active_stage.json")
    worker_state = _read_json(run_dir / "worker_state.json")
    stage_age_seconds: float | None = None
    active_stage_path = run_dir / "active_stage.json"
    if active_stage_path.exists():
        stage_age_seconds = round(time.time() - active_stage_path.stat().st_mtime, 3)
    else:
        worker_state_path = run_dir / "worker_state.json"
        if worker_state_path.exists() and worker_state.get("status") == "running":
            stage_age_seconds = round(time.time() - worker_state_path.stat().st_mtime, 3)
    return {
        "active_stage": active_stage or None,
        "active_stage_age_seconds": stage_age_seconds,
        "worker_pid": worker_state.get("worker_pid") or worker_state.get("pid"),
        "worker_status": worker_state.get("status"),
        "worker_loaded_model": worker_state.get("model_loaded") is True,
    }


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
    processes: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2 or not parts[0]:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        used_gpu_memory_mib: int | None
        try:
            used_gpu_memory_mib = int(parts[1])
        except ValueError:
            used_gpu_memory_mib = None
        processes.append(
            {
                "pid": pid,
                "used_gpu_memory_mib": used_gpu_memory_mib,
            }
        )
    return processes


def status_timestamp_now() -> str:
    return datetime.now(UTC).isoformat()


def write_status_snapshot(*, output_root: str | Path, status: dict[str, object]) -> None:
    output_root_path = Path(output_root)
    status_path = output_root_path / "batch_status.json"
    write_json_dict(status_path, status)
