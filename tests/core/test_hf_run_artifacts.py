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


def test_normalize_stale_running_run_preserves_existing_state_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "status": "running",
                "replication": 3,
                "condition_bits": "10101",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "worker_state.json").write_text(
        json.dumps({"status": "running", "pid": 999999}),
        encoding="utf-8",
    )

    normalize_stale_running_run(run_dir)

    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert state == {
        "status": "failed",
        "replication": 3,
        "condition_bits": "10101",
    }


def test_normalize_stale_running_run_keeps_live_persistent_worker(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "state.json").write_text('{"status": "running"}', encoding="utf-8")
    (run_dir / "worker_state.json").write_text(
        json.dumps({"status": "running", "worker_pid": os.getpid()}),
        encoding="utf-8",
    )

    error = normalize_stale_running_run(run_dir)

    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert error is None
    assert state["status"] == "running"


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


def test_collect_batch_status_requires_boolean_true_for_worker_loaded_model(
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
                "model_loaded": "false",
            }
        ),
        encoding="utf-8",
    )
    (output_root / "batch_status.json").write_text(
        json.dumps({"total_runs": 1, "current_run": {"algorithm": "algo1"}}),
        encoding="utf-8",
    )

    status = collect_batch_status(output_root)

    assert status["worker_loaded_model"] is False


def test_collect_batch_status_ignores_malformed_total_runs_when_run_tree_is_empty(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "results"
    output_root.mkdir()
    (output_root / "batch_status.json").write_text(
        json.dumps(
            {
                "total_runs": "false",
                "current_run": {"algorithm": "algo1"},
            }
        ),
        encoding="utf-8",
    )

    status = collect_batch_status(output_root)

    assert status["total_runs"] == 0
    assert status["pending_count"] == 0
    assert status["percent_complete"] == 0.0


def test_collect_batch_status_filters_inactive_models_from_runtime_config(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "results"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "runtime_config.yaml").write_text(
        """
run:
  provider: hf-transformers
  output_root: /tmp/results
  replications: 1
runtime:
  seed: 7
  temperature: 0.0
  quantization: none
  device_policy: cuda-only
  context_policy:
    prompt_truncation: forbid
  max_new_tokens_by_schema:
    edge_list: 128
models:
  chat_models:
    - mistralai/Ministral-3-8B-Instruct-2512
    - Qwen/Qwen3.5-9B
  embedding_model: Qwen/Qwen3-Embedding-0.6B
""".strip()
        + "\n",
        encoding="utf-8",
    )

    active_finished_dir = (
        output_root
        / "runs"
        / "algo1"
        / "Qwen__Qwen3.5-9B"
        / "greedy"
        / "sg1_sg2"
        / "00000"
        / "rep_00"
    )
    active_running_dir = (
        output_root
        / "runs"
        / "algo1"
        / "mistralai__Ministral-3-8B-Instruct-2512"
        / "greedy"
        / "sg1_sg2"
        / "00001"
        / "rep_01"
    )
    inactive_failed_dir = (
        output_root
        / "runs"
        / "algo1"
        / "allenai__Olmo-3-7B-Instruct"
        / "greedy"
        / "sg1_sg2"
        / "00010"
        / "rep_02"
    )
    for directory in [active_finished_dir, active_running_dir, inactive_failed_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    (active_finished_dir / "state.json").write_text('{"status": "finished"}', encoding="utf-8")
    (active_finished_dir / "summary.json").write_text('{"status": "finished"}', encoding="utf-8")
    (active_running_dir / "state.json").write_text('{"status": "running"}', encoding="utf-8")
    (inactive_failed_dir / "state.json").write_text('{"status": "failed"}', encoding="utf-8")
    (inactive_failed_dir / "error.json").write_text('{"message": "boom"}', encoding="utf-8")
    (output_root / "batch_status.json").write_text(
        json.dumps({"total_runs": 3, "current_run": {"algorithm": "algo1"}}),
        encoding="utf-8",
    )

    status = collect_batch_status(output_root)

    assert status["finished_count"] == 1
    assert status["running_count"] == 1
    assert status["failed_count"] == 0
    assert status["pending_count"] == 0
    assert status["total_runs"] == 2


def test_collect_batch_status_respects_shard_manifest_identity_scope(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "results"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "runtime_config.yaml").write_text(
        """
run:
  provider: hf-transformers
  output_root: /tmp/results
  replications: 1
runtime:
  seed: 7
  temperature: 0.0
  quantization: none
  device_policy: cuda-only
  context_policy:
    prompt_truncation: forbid
  max_new_tokens_by_schema:
    edge_list: 128
models:
  chat_models:
    - mistralai/Ministral-3-8B-Instruct-2512
    - Qwen/Qwen3.5-9B
  embedding_model: Qwen/Qwen3-Embedding-0.6B
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (output_root / "shard_manifest.json").write_text(
        json.dumps(
            {
                "shard_count": 1,
                "shard_index": 0,
                "identities": [
                    {
                        "algorithm": "algo2",
                        "model": "mistralai/Ministral-3-8B-Instruct-2512",
                        "condition_label": "beam_num_beams_2",
                        "pair_name": "sg3_sg1",
                        "condition_bits": "000001",
                        "replication": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    scoped_dir = (
        output_root
        / "runs"
        / "algo2"
        / "mistralai__Ministral-3-8B-Instruct-2512"
        / "beam_num_beams_2"
        / "sg3_sg1"
        / "000001"
        / "rep_01"
    )
    ignored_dir = (
        output_root
        / "runs"
        / "algo1"
        / "Qwen__Qwen3.5-9B"
        / "greedy"
        / "sg1_sg2"
        / "00000"
        / "rep_00"
    )
    scoped_dir.mkdir(parents=True, exist_ok=True)
    ignored_dir.mkdir(parents=True, exist_ok=True)
    (scoped_dir / "state.json").write_text('{"status": "running"}', encoding="utf-8")
    (ignored_dir / "state.json").write_text('{"status": "finished"}', encoding="utf-8")
    (ignored_dir / "summary.json").write_text('{"status": "finished"}', encoding="utf-8")
    (output_root / "batch_status.json").write_text(
        json.dumps({"total_runs": 2, "current_run": {"algorithm": "algo2"}}),
        encoding="utf-8",
    )

    status = collect_batch_status(output_root)

    assert status["total_runs"] == 1
    assert status["running_count"] == 1
    assert status["finished_count"] == 0
    assert status["pending_count"] == 0
