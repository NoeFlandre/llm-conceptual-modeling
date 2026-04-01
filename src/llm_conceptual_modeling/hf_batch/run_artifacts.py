from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

VOLATILE_RUN_FILENAMES = (
    "active_stage.json",
    "error.json",
    "raw_response.json",
    "runtime.json",
    "worker_result.json",
    "worker_spec.json",
    "worker_state.json",
)


def clear_retry_artifacts(run_dir: Path) -> None:
    for name in VOLATILE_RUN_FILENAMES:
        artifact_path = run_dir / name
        if artifact_path.exists():
            artifact_path.unlink()


def normalize_stale_running_run(run_dir: Path) -> dict[str, object] | None:
    state = read_json(run_dir / "state.json")
    if state.get("status") != "running":
        return None
    worker_result_path = run_dir / "worker_result.json"
    if worker_result_path.exists():
        return None
    worker_state = read_json(run_dir / "worker_state.json")
    worker_pid = worker_state.get("pid")
    if _is_live_worker_pid(worker_pid):
        return None

    active_stage = read_json(run_dir / "active_stage.json")
    error_payload = {
        "type": "StaleRunState",
        "message": (
            "Found stale running state with no live worker or worker result artifact. "
            "Marking the run as failed so resume can retry safely."
        ),
        "status": "failed",
        "worker_pid": worker_pid,
        "active_stage": active_stage or None,
        "updated_at": _timestamp_now(),
    }
    write_json(run_dir / "error.json", error_payload)
    write_json(run_dir / "state.json", {"status": "failed"})
    return error_payload


def write_smoke_verdict(
    *,
    output_root: Path,
    run_dir: Path,
    spec_identity: dict[str, object],
    status: str,
    failure_type: str | None = None,
    failure_message: str | None = None,
    worker_loaded_model: bool = False,
) -> dict[str, object]:
    verdict: dict[str, object] = {
        "status": status,
        "failure_type": failure_type,
        "failure_message": failure_message,
        "worker_loaded_model": worker_loaded_model,
        "runtime_snapshot_path": _runtime_snapshot_reference(),
        "run_dir": str(run_dir),
        "summary_path": str(run_dir / "summary.json"),
        "error_path": str(run_dir / "error.json"),
        "spec": spec_identity,
        "updated_at": _timestamp_now(),
    }
    write_json(output_root / "smoke_verdict.json", verdict)
    return verdict


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _runtime_snapshot_reference() -> str | None:
    snapshot_path = Path.cwd() / ".bootstrap-runtime.json"
    if not snapshot_path.exists():
        return None
    return str(snapshot_path)


def _timestamp_now() -> str:
    return datetime.now(UTC).isoformat()


def _is_live_worker_pid(raw_pid: object) -> bool:
    if not isinstance(raw_pid, int) or raw_pid <= 0:
        return False
    try:
        os.kill(raw_pid, 0)
    except OSError:
        return False
    return True
