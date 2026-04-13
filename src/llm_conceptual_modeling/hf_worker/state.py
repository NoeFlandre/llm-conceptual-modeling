from __future__ import annotations

import json
from pathlib import Path

from llm_conceptual_modeling.common.io import write_json_dict


def read_worker_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def update_worker_state(path: Path, updates: dict[str, object]) -> dict[str, object]:
    payload = read_worker_state(path)
    payload.update(updates)
    write_json_dict(path, payload)
    return payload


def mark_worker_prefetching_model(
    path: Path,
    *,
    worker_pid: int,
    requests_served_by_process: int,
    timestamp: str,
) -> dict[str, object]:
    return update_worker_state(
        path,
        _running_worker_state_update(timestamp)
        | {
            "phase": "prefetching_model",
            "worker_pid": worker_pid,
            "model_loaded": False,
            "requests_served_by_process": requests_served_by_process,
        },
    )


def mark_worker_loading_model(
    path: Path,
    *,
    worker_pid: int,
    requests_served_by_process: int,
    timestamp: str,
) -> dict[str, object]:
    return mark_worker_prefetching_model(
        path,
        worker_pid=worker_pid,
        requests_served_by_process=requests_served_by_process,
        timestamp=timestamp,
    )


def mark_worker_ready_for_execution(
    path: Path,
    *,
    timestamp: str,
) -> dict[str, object]:
    return update_worker_state(
        path,
        _running_worker_state_update(timestamp)
        | {
            "phase": "executing_algorithm",
            "model_loaded": True,
            "model_loaded_at": timestamp,
        },
    )


def worker_has_started_stage_execution(worker_state: dict[str, object]) -> bool:
    if worker_state.get("model_loaded") is True:
        return True
    phase = worker_state.get("phase")
    return isinstance(phase, str) and phase == "executing_algorithm"


def _running_worker_state_update(timestamp: str) -> dict[str, object]:
    return {
        "status": "running",
        "updated_at": timestamp,
    }
