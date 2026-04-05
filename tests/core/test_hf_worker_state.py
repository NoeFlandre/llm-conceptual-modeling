from __future__ import annotations

import json
from pathlib import Path

from llm_conceptual_modeling.hf_worker_state import (
    mark_worker_prefetching_model,
    mark_worker_ready_for_execution,
    update_worker_state,
    worker_has_started_stage_execution,
)


def test_mark_worker_prefetching_model_sets_prefetch_phase(tmp_path: Path) -> None:
    worker_state_path = tmp_path / "worker_state.json"

    mark_worker_prefetching_model(
        worker_state_path,
        worker_pid=1234,
        requests_served_by_process=2,
        timestamp="2026-04-03T00:00:00+00:00",
    )

    payload = json.loads(worker_state_path.read_text(encoding="utf-8"))
    assert payload == {
        "model_loaded": False,
        "phase": "prefetching_model",
        "requests_served_by_process": 2,
        "status": "running",
        "updated_at": "2026-04-03T00:00:00+00:00",
        "worker_pid": 1234,
    }


def test_mark_worker_ready_for_execution_preserves_existing_fields(tmp_path: Path) -> None:
    worker_state_path = tmp_path / "worker_state.json"
    update_worker_state(
        worker_state_path,
        {
            "status": "running",
            "phase": "prefetching_model",
            "worker_pid": 2222,
            "requests_served_by_process": 4,
            "model_loaded": False,
        },
    )

    mark_worker_ready_for_execution(
        worker_state_path,
        timestamp="2026-04-03T00:00:05+00:00",
    )

    payload = json.loads(worker_state_path.read_text(encoding="utf-8"))
    assert payload["worker_pid"] == 2222
    assert payload["requests_served_by_process"] == 4
    assert payload["phase"] == "executing_algorithm"
    assert payload["model_loaded"] is True
    assert payload["model_loaded_at"] == "2026-04-03T00:00:05+00:00"
    assert payload["updated_at"] == "2026-04-03T00:00:05+00:00"


def test_worker_has_started_stage_execution_uses_phase_and_model_flag() -> None:
    assert worker_has_started_stage_execution({"model_loaded": True}) is True
    assert worker_has_started_stage_execution({"model_loaded": "false"}) is False
    assert worker_has_started_stage_execution({"phase": "executing_algorithm"}) is True
    assert worker_has_started_stage_execution({"phase": "prefetching_model"}) is False
