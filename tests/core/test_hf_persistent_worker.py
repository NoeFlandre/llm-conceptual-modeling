from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_conceptual_modeling.common.hf_transformers import DecodingConfig, RuntimeProfile
from llm_conceptual_modeling.hf_batch_types import HFRunSpec
from llm_conceptual_modeling.hf_persistent_worker import PersistentHFWorkerSession
from llm_conceptual_modeling.hf_subprocess import MonitoredCommandTimeout


class _FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode


def _spec(pair_name: str) -> HFRunSpec:
    return HFRunSpec(
        algorithm="algo1",
        model="allenai/Olmo-3-7B-Instruct",
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        decoding=DecodingConfig(algorithm="greedy"),
        replication=0,
        pair_name=pair_name,
        condition_bits="00000",
        condition_label="greedy",
        prompt_factors={},
        raw_context={"pair_name": pair_name, "Repetition": 0},
        input_payload={
            "subgraph1": [("alpha", "beta")],
            "subgraph2": [("gamma", "delta")],
            "graph": [("alpha", "gamma")],
        },
        runtime_profile=RuntimeProfile(
            device="cuda",
            dtype="bfloat16",
            quantization="none",
            supports_thinking_toggle=False,
            context_limit=4096,
        ),
        context_policy={"generation_timeout_seconds": 20},
    )


def test_persistent_session_reuses_one_worker_process_for_two_specs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_dir = tmp_path / "queue"
    spawn_pids: list[int] = []
    session = PersistentHFWorkerSession(queue_dir=queue_dir, worker_python="/tmp/fake-python")

    def fake_spawn(self) -> _FakeProcess:
        pid = 1000 + len(spawn_pids)
        spawn_pids.append(pid)
        return _FakeProcess(pid)

    def fake_wait(
        self,
        *,
        process,
        request_path,
        result_json_path,
        run_dir,
        startup_timeout_seconds,
        stage_timeout_seconds,
    ):
        _ = (request_path, startup_timeout_seconds, stage_timeout_seconds)
        (run_dir / "worker_state.json").write_text(
            json.dumps({"status": "finished", "worker_pid": process.pid}),
            encoding="utf-8",
        )
        result_json_path.write_text(
            json.dumps(
                {
                    "ok": True,
                    "runtime_result": {
                        "raw_row": {},
                        "runtime": {},
                        "raw_response": "{}",
                    },
                }
            ),
            encoding="utf-8",
        )
        return {"raw_row": {}, "runtime": {}, "raw_response": "{}"}

    monkeypatch.setattr(PersistentHFWorkerSession, "_spawn_worker_process", fake_spawn)
    monkeypatch.setattr(PersistentHFWorkerSession, "_wait_for_result", fake_wait)

    first = session.run_spec(spec=_spec("sg1_sg2"), run_dir=tmp_path / "run1")
    second = session.run_spec(spec=_spec("sg2_sg3"), run_dir=tmp_path / "run2")

    assert first["raw_response"] == "{}"
    assert second["raw_response"] == "{}"
    assert spawn_pids == [1000]


def test_persistent_session_restarts_after_request_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_dir = tmp_path / "queue"
    spawn_pids: list[int] = []
    terminated_pids: list[int] = []
    session = PersistentHFWorkerSession(queue_dir=queue_dir, worker_python="/tmp/fake-python")

    def fake_spawn(self) -> _FakeProcess:
        pid = 2000 + len(spawn_pids)
        spawn_pids.append(pid)
        return _FakeProcess(pid)

    def fake_wait(
        self,
        *,
        process,
        request_path,
        result_json_path,
        run_dir,
        startup_timeout_seconds,
        stage_timeout_seconds,
    ):
        _ = (
            request_path,
            result_json_path,
            run_dir,
            startup_timeout_seconds,
            stage_timeout_seconds,
        )
        if len(spawn_pids) == 1:
            raise MonitoredCommandTimeout("stage", 20.0)
        return {"raw_row": {}, "runtime": {}, "raw_response": "{}"}

    def fake_terminate(self, process: _FakeProcess) -> None:
        terminated_pids.append(process.pid)
        process.returncode = -15

    monkeypatch.setattr(PersistentHFWorkerSession, "_spawn_worker_process", fake_spawn)
    monkeypatch.setattr(PersistentHFWorkerSession, "_wait_for_result", fake_wait)
    monkeypatch.setattr(PersistentHFWorkerSession, "_terminate_process", fake_terminate)

    with pytest.raises(MonitoredCommandTimeout):
        session.run_spec(spec=_spec("sg1_sg2"), run_dir=tmp_path / "run1")

    result = session.run_spec(spec=_spec("sg2_sg3"), run_dir=tmp_path / "run2")

    assert result["raw_response"] == "{}"
    assert spawn_pids == [2000, 2001]
    assert terminated_pids == [2000]
    assert sorted(queue_dir.glob("*.request.json")) == []


def test_persistent_session_recycles_worker_after_request_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_dir = tmp_path / "queue"
    spawn_pids: list[int] = []
    terminated_pids: list[int] = []
    session = PersistentHFWorkerSession(
        queue_dir=queue_dir,
        worker_python="/tmp/fake-python",
        max_requests_per_process=2,
    )

    def fake_spawn(self) -> _FakeProcess:
        pid = 3000 + len(spawn_pids)
        spawn_pids.append(pid)
        return _FakeProcess(pid)

    def fake_wait(
        self,
        *,
        process,
        request_path,
        result_json_path,
        run_dir,
        startup_timeout_seconds,
        stage_timeout_seconds,
    ):
        _ = (
            process,
            request_path,
            result_json_path,
            run_dir,
            startup_timeout_seconds,
            stage_timeout_seconds,
        )
        return {"raw_row": {}, "runtime": {}, "raw_response": "{}"}

    def fake_terminate(self, process: _FakeProcess) -> None:
        terminated_pids.append(process.pid)
        process.returncode = 0

    monkeypatch.setattr(PersistentHFWorkerSession, "_spawn_worker_process", fake_spawn)
    monkeypatch.setattr(PersistentHFWorkerSession, "_wait_for_result", fake_wait)
    monkeypatch.setattr(PersistentHFWorkerSession, "_terminate_process", fake_terminate)

    session.run_spec(spec=_spec("sg1_sg2"), run_dir=tmp_path / "run1")
    session.run_spec(spec=_spec("sg2_sg3"), run_dir=tmp_path / "run2")
    session.run_spec(spec=_spec("sg3_sg1"), run_dir=tmp_path / "run3")

    assert spawn_pids == [3000, 3001]
    assert terminated_pids == [3000]


def test_persistent_session_retries_broken_pipe_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_dir = tmp_path / "queue"
    spawn_pids: list[int] = []
    terminated_pids: list[int] = []
    session = PersistentHFWorkerSession(queue_dir=queue_dir, worker_python="/tmp/fake-python")

    def fake_spawn(self) -> _FakeProcess:
        pid = 4000 + len(spawn_pids)
        spawn_pids.append(pid)
        return _FakeProcess(pid)

    attempts = {"count": 0}

    def fake_wait(
        self,
        *,
        process,
        request_path,
        result_json_path,
        run_dir,
        startup_timeout_seconds,
        stage_timeout_seconds,
    ):
        _ = (
            process,
            request_path,
            result_json_path,
            run_dir,
            startup_timeout_seconds,
            stage_timeout_seconds,
        )
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("RuntimeError: BrokenPipeError: [Errno 32] Broken pipe")
        return {"raw_row": {}, "runtime": {}, "raw_response": "{}"}

    def fake_terminate(self, process: _FakeProcess) -> None:
        terminated_pids.append(process.pid)
        process.returncode = -15

    monkeypatch.setattr(PersistentHFWorkerSession, "_spawn_worker_process", fake_spawn)
    monkeypatch.setattr(PersistentHFWorkerSession, "_wait_for_result", fake_wait)
    monkeypatch.setattr(PersistentHFWorkerSession, "_terminate_process", fake_terminate)

    result = session.run_spec(spec=_spec("sg1_sg2"), run_dir=tmp_path / "run1")

    assert result["raw_response"] == "{}"
    assert attempts["count"] == 2
    assert spawn_pids == [4000, 4001]
    assert terminated_pids == [4000]
