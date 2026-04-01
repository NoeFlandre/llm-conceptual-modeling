import json
import os
import time
from pathlib import Path

from llm_conceptual_modeling.hf_batch.monitoring import collect_batch_status
from llm_conceptual_modeling.hf_batch.run_artifacts import (
    clear_retry_artifacts,
    normalize_stale_running_run,
)


def test_normalize_stale_running_run_marks_failed_when_worker_is_gone(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "state.json").write_text('{"status": "running"}', encoding="utf-8")
    (run_dir / "worker_state.json").write_text(
        json.dumps({"status": "running", "pid": 999999}),
        encoding="utf-8",
    )
    (run_dir / "active_stage.json").write_text(
        json.dumps({"status": "running", "schema_name": "edge_list"}),
        encoding="utf-8",
    )

    error = normalize_stale_running_run(run_dir)

    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    persisted_error = json.loads((run_dir / "error.json").read_text(encoding="utf-8"))
    assert error is not None
    assert state["status"] == "failed"
    assert persisted_error["type"] == "StaleRunState"
    assert persisted_error["active_stage"]["schema_name"] == "edge_list"


def test_clear_retry_artifacts_preserves_stage_cache_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stages"
    stage_dir.mkdir(parents=True)
    volatile_path = run_dir / "worker_state.json"
    stage_cache_path = stage_dir / "algo1_edge_generation.json"
    volatile_path.write_text("{}", encoding="utf-8")
    stage_cache_path.write_text('{"candidate_edges": []}', encoding="utf-8")

    clear_retry_artifacts(run_dir)

    assert not volatile_path.exists()
    assert stage_cache_path.exists()


def test_collect_batch_status_uses_worker_state_age_when_stage_file_is_missing(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "results"
    run_dir = output_root / "runs" / "algo1" / "model" / "greedy" / "sg1_sg2" / "00000" / "rep_00"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "state.json").write_text('{"status": "running"}', encoding="utf-8")
    (run_dir / "worker_state.json").write_text(
        json.dumps(
            {
                "status": "running",
                "worker_pid": 1234,
                "updated_at": "2026-03-31T00:01:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    old_time = time.time() - 5
    os.utime(run_dir / "worker_state.json", (old_time, old_time))
    (output_root / "batch_status.json").write_text(
        json.dumps({"total_runs": 1, "current_run": {"algorithm": "algo1"}}),
        encoding="utf-8",
    )

    status = collect_batch_status(output_root)

    assert status["running_count"] == 1
    assert status["active_stage"] is None
    assert isinstance(status["active_stage_age_seconds"], float)
    assert status["active_stage_age_seconds"] >= 0.0
