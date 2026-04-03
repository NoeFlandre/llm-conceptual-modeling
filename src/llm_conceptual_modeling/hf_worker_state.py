from __future__ import annotations

import json
from pathlib import Path


def read_worker_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def update_worker_state(path: Path, updates: dict[str, object]) -> dict[str, object]:
    payload = read_worker_state(path)
    payload.update(updates)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def mark_worker_loading_model(
    path: Path,
    *,
    worker_pid: int,
    requests_served_by_process: int,
    timestamp: str,
) -> dict[str, object]:
    return update_worker_state(
        path,
        {
            "status": "running",
            "phase": "loading_model",
            "worker_pid": worker_pid,
            "model_loaded": False,
            "requests_served_by_process": requests_served_by_process,
            "updated_at": timestamp,
        },
    )


def mark_worker_ready_for_execution(
    path: Path,
    *,
    timestamp: str,
) -> dict[str, object]:
    return update_worker_state(
        path,
        {
            "status": "running",
            "phase": "executing_algorithm",
            "model_loaded": True,
            "model_loaded_at": timestamp,
            "updated_at": timestamp,
        },
    )


def worker_has_started_stage_execution(worker_state: dict[str, object]) -> bool:
    if bool(worker_state.get("model_loaded")):
        return True
    phase = worker_state.get("phase")
    return isinstance(phase, str) and phase == "executing_algorithm"
