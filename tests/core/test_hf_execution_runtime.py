from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_conceptual_modeling.common.hf_transformers import DecodingConfig, RuntimeProfile
from llm_conceptual_modeling.hf_execution.subprocess import MonitoredCommandTimeout
from llm_conceptual_modeling.hf_experiments import HFRunSpec


def _runtime_profile() -> RuntimeProfile:
    return RuntimeProfile(
        device="cuda",
        dtype="bfloat16",
        quantization="none",
        supports_thinking_toggle=False,
        context_limit=4096,
    )


def _algo1_spec(*, context_policy: dict[str, object] | None = None) -> HFRunSpec:
    return HFRunSpec(
        algorithm="algo1",
        model="allenai/Olmo-3-7B-Instruct",
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        decoding=DecodingConfig(algorithm="greedy", temperature=0.0),
        replication=0,
        pair_name="sg1_sg2",
        condition_bits="00000",
        condition_label="greedy",
        prompt_factors={},
        raw_context={"pair_name": "sg1_sg2", "Repetition": 0},
        input_payload={
            "subgraph1": [("alpha", "beta")],
            "subgraph2": [("gamma", "delta")],
            "graph": [("alpha", "gamma")],
        },
        runtime_profile=_runtime_profile(),
        context_policy=context_policy or {},
    )


def _contrastive_spec(*, context_policy: dict[str, object] | None = None) -> HFRunSpec:
    return HFRunSpec(
        algorithm="algo3",
        model="mistralai/Ministral-3-8B-Instruct-2512",
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        decoding=DecodingConfig(algorithm="contrastive", penalty_alpha=0.8, top_k=4),
        replication=0,
        pair_name="subgraph_1_to_subgraph_3",
        condition_bits="1000",
        condition_label="contrastive_penalty_alpha_0.8",
        prompt_factors={},
        raw_context={"pair_name": "subgraph_1_to_subgraph_3", "Repetition": 0},
        input_payload={"source_graph": [], "target_graph": []},
        runtime_profile=_runtime_profile(),
        context_policy=context_policy or {},
    )


def test_resolve_worker_process_mode_defaults_to_ephemeral() -> None:
    from llm_conceptual_modeling.hf_execution_runtime import resolve_worker_process_mode

    assert resolve_worker_process_mode(None) == "ephemeral"


def test_resolve_max_requests_per_worker_process_rejects_non_positive() -> None:
    from llm_conceptual_modeling.hf_execution_runtime import (
        resolve_max_requests_per_worker_process,
    )

    with pytest.raises(ValueError, match="must be positive"):
        resolve_max_requests_per_worker_process({"max_requests_per_worker_process": 0})


@pytest.mark.parametrize(
    "error",
    [
        {
            "type": "RuntimeError",
            "message": "ValueError: Structured response returned an empty edge target",
        },
        {
            "type": "RuntimeError",
            "message": (
                "OutOfMemoryError: CUDA out of memory. Tried to allocate 374.00 MiB."
            ),
        },
        {
            "type": "RuntimeError",
            "message": "JSONDecodeError: Expecting value: line 1 column 1 (char 0)",
        },
        {
            "type": "RuntimeError",
            "message": "ModuleNotFoundError: No module named 'llm_conceptual_modeling'",
        },
    ],
)
def test_is_retryable_worker_error_recognizes_structural_and_infrastructure_wrappers(
    error: dict[str, str],
) -> None:
    from llm_conceptual_modeling.hf_execution_runtime import is_retryable_worker_error

    assert is_retryable_worker_error(error)


def test_run_local_hf_spec_persistent_mode_reuses_existing_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from llm_conceptual_modeling.hf_execution_runtime import run_local_hf_spec

    spec = HFRunSpec(
        algorithm="algo1",
        model="allenai/Olmo-3-7B-Instruct",
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        decoding=DecodingConfig(algorithm="greedy"),
        replication=0,
        pair_name="sg1_sg2",
        condition_bits="00000",
        condition_label="greedy",
        prompt_factors={},
        raw_context={"pair_name": "sg1_sg2", "Repetition": 0},
        input_payload={
            "subgraph1": [("alpha", "beta")],
            "subgraph2": [("gamma", "delta")],
            "graph": [("alpha", "gamma")],
        },
        runtime_profile=_runtime_profile(),
        context_policy={
            "worker_process_mode": "persistent",
            "max_requests_per_worker_process": 9,
        },
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    output_root = tmp_path / "results"
    persistent_sessions: dict[str, object] = {}
    session_calls: list[tuple[str, Path]] = []

    class FakeSession:
        def __init__(
            self,
            *,
            queue_dir: Path,
            worker_python: str,
            max_requests_per_process: int | None = None,
        ) -> None:
            assert queue_dir == output_root / "worker-queues" / "allenai__Olmo-3-7B-Instruct"
            assert worker_python
            assert max_requests_per_process == 9

        def run_spec(self, *, spec: HFRunSpec, run_dir: Path) -> dict[str, object]:
            session_calls.append((spec.pair_name, run_dir))
            return {
                "raw_row": {"pair_name": spec.pair_name},
                "runtime": {},
                "raw_response": "{}",
            }

    monkeypatch.setattr(
        "llm_conceptual_modeling.hf_execution_runtime.PersistentHFWorkerSession",
        FakeSession,
    )

    first_result = run_local_hf_spec(
        spec=spec,
        run_dir=run_dir,
        output_root=output_root,
        persistent_sessions=persistent_sessions,
    )
    second_result = run_local_hf_spec(
        spec=spec,
        run_dir=run_dir,
        output_root=output_root,
        persistent_sessions=persistent_sessions,
    )

    assert first_result["raw_row"]["pair_name"] == "sg1_sg2"
    assert second_result["raw_row"]["pair_name"] == "sg1_sg2"
    assert session_calls == [("sg1_sg2", run_dir), ("sg1_sg2", run_dir)]
    assert len(persistent_sessions) == 1


def test_run_local_hf_spec_uses_persistent_session_for_contrastive_runs_in_persistent_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from llm_conceptual_modeling.hf_execution_runtime import run_local_hf_spec

    spec = _contrastive_spec(
        context_policy={
            "worker_process_mode": "persistent",
            "max_requests_per_worker_process": 9,
        }
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    output_root = tmp_path / "results"
    persistent_sessions: dict[str, object] = {}
    session_calls: list[tuple[str, Path]] = []

    class FakeSession:
        def __init__(
            self,
            *,
            queue_dir: Path,
            worker_python: str,
            max_requests_per_process: int | None = None,
        ) -> None:
            assert queue_dir == output_root / "worker-queues" / "mistralai__Ministral-3-8B-Instruct-2512"
            assert worker_python
            assert max_requests_per_process == 9

        def run_spec(self, *, spec: HFRunSpec, run_dir: Path) -> dict[str, object]:
            session_calls.append((spec.pair_name, run_dir))
            return {
                "raw_row": {"pair_name": spec.pair_name},
                "runtime": {},
                "raw_response": "{}",
            }

    monkeypatch.setattr(
        "llm_conceptual_modeling.hf_execution_runtime.PersistentHFWorkerSession",
        FakeSession,
    )

    result = run_local_hf_spec(
        spec=spec,
        run_dir=run_dir,
        output_root=output_root,
        persistent_sessions=persistent_sessions,
    )

    assert result["raw_row"]["pair_name"] == "subgraph_1_to_subgraph_3"
    assert session_calls == [("subgraph_1_to_subgraph_3", run_dir)]
    assert list(persistent_sessions) == ["mistralai/Ministral-3-8B-Instruct-2512"]


def test_run_local_hf_spec_closes_other_model_sessions_before_reusing_persistent_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from llm_conceptual_modeling.hf_execution_runtime import run_local_hf_spec

    spec = _algo1_spec(
        context_policy={
            "worker_process_mode": "persistent",
            "max_requests_per_worker_process": 9,
        }
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    output_root = tmp_path / "results"
    closed_sessions: list[str] = []

    class ExistingSession:
        def close(self) -> None:
            closed_sessions.append("old-model")

    class FakeSession:
        def __init__(
            self,
            *,
            queue_dir: Path,
            worker_python: str,
            max_requests_per_process: int | None = None,
        ) -> None:
            assert queue_dir == output_root / "worker-queues" / "allenai__Olmo-3-7B-Instruct"
            assert worker_python
            assert max_requests_per_process == 9

        def run_spec(self, *, spec: HFRunSpec, run_dir: Path) -> dict[str, object]:
            return {
                "raw_row": {"pair_name": spec.pair_name},
                "runtime": {},
                "raw_response": "{}",
            }

    monkeypatch.setattr(
        "llm_conceptual_modeling.hf_execution_runtime.PersistentHFWorkerSession",
        FakeSession,
    )
    persistent_sessions: dict[str, object] = {
        "Qwen/Qwen3.5-9B": ExistingSession(),
    }

    result = run_local_hf_spec(
        spec=spec,
        run_dir=run_dir,
        output_root=output_root,
        persistent_sessions=persistent_sessions,
    )

    assert result["raw_row"]["pair_name"] == "sg1_sg2"
    assert closed_sessions == ["old-model"]
    assert list(persistent_sessions) == ["allenai/Olmo-3-7B-Instruct"]


def test_run_local_hf_spec_keeps_matching_model_session_for_contrastive_persistent_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from llm_conceptual_modeling.hf_execution_runtime import run_local_hf_spec

    spec = _contrastive_spec(
        context_policy={
            "worker_process_mode": "persistent",
            "max_requests_per_worker_process": 9,
        }
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    output_root = tmp_path / "results"
    closed_sessions: list[str] = []
    session_calls: list[tuple[str, Path]] = []

    class ExistingSession:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            closed_sessions.append(self.name)

        def run_spec(self, *, spec: HFRunSpec, run_dir: Path) -> dict[str, object]:
            session_calls.append((spec.pair_name, run_dir))
            return {
                "raw_row": {"pair_name": spec.pair_name},
                "runtime": {},
                "raw_response": "{}",
            }

    monkeypatch.setattr(
        "llm_conceptual_modeling.hf_execution_runtime.PersistentHFWorkerSession",
        ExistingSession,
    )

    persistent_sessions: dict[str, object] = {
        "Qwen/Qwen3.5-9B": ExistingSession("qwen"),
        "mistralai/Ministral-3-8B-Instruct-2512": ExistingSession("mistral"),
    }

    result = run_local_hf_spec(
        spec=spec,
        run_dir=run_dir,
        output_root=output_root,
        persistent_sessions=persistent_sessions,
    )

    assert result["raw_row"]["pair_name"] == "subgraph_1_to_subgraph_3"
    assert session_calls == [("subgraph_1_to_subgraph_3", run_dir)]
    assert closed_sessions == ["qwen"]
    assert list(persistent_sessions) == ["mistralai/Ministral-3-8B-Instruct-2512"]


def test_run_local_hf_spec_subprocess_retries_retryable_structural_validation_failure(
    tmp_path: Path,
) -> None:
    from llm_conceptual_modeling.hf_execution_runtime import run_local_hf_spec_subprocess

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    spec = _algo1_spec()
    attempts = {"count": 0}

    def fake_run_monitored_command(**kwargs):
        attempts["count"] += 1
        result_json_path = Path(kwargs["command"][6])
        result_json_path.write_text(
            json.dumps(
                {
                    "ok": True,
                    "runtime_result": {
                        "raw_row": {
                            "Result": "[('alpha', 'gamma')]",
                            "graph": "[('alpha', 'gamma')]",
                            "subgraph1": "[('alpha', 'beta')]",
                            "subgraph2": "[('gamma', 'delta')]",
                        },
                        "runtime": {"thinking_mode_supported": False},
                        "raw_response": "[]",
                    },
                }
            ),
            encoding="utf-8",
        )
        return type("Completed", (), {"stdout": "", "stderr": ""})()

    def fake_validate_runtime_result(*, algorithm: str, raw_row: dict[str, object]) -> None:
        if attempts["count"] == 1:
            raise ValueError(
                "Structurally invalid algo1 result: verified edge list is empty."
            )

    actual = run_local_hf_spec_subprocess(
        spec=spec,
        run_dir=run_dir,
        run_monitored_command_fn=fake_run_monitored_command,
        validate_runtime_result_fn=fake_validate_runtime_result,
    )

    assert attempts["count"] == 2
    assert actual["raw_row"]["Result"] == "[('alpha', 'gamma')]"


def test_run_local_hf_spec_subprocess_retries_monitored_command_timeout(
    tmp_path: Path,
) -> None:
    from llm_conceptual_modeling.hf_execution_runtime import run_local_hf_spec_subprocess

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    spec = _algo1_spec()
    attempts = {"count": 0}

    def fake_run_monitored_command(**kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise MonitoredCommandTimeout("stage", 180.0)
        result_json_path = Path(kwargs["command"][6])
        result_json_path.write_text(
            json.dumps(
                {
                    "ok": True,
                    "runtime_result": {
                        "raw_row": {
                            "Result": "[('alpha', 'gamma')]",
                            "graph": "[('alpha', 'gamma')]",
                            "subgraph1": "[('alpha', 'beta')]",
                            "subgraph2": "[('gamma', 'delta')]",
                        },
                        "runtime": {"thinking_mode_supported": False},
                        "raw_response": "[]",
                    },
                }
            ),
            encoding="utf-8",
        )
        return type("Completed", (), {"stdout": "", "stderr": ""})()

    actual = run_local_hf_spec_subprocess(
        spec=spec,
        run_dir=run_dir,
        run_monitored_command_fn=fake_run_monitored_command,
    )

    assert attempts["count"] == 2
    assert actual["raw_row"]["Result"] == "[('alpha', 'gamma')]"
