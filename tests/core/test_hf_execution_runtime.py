from __future__ import annotations

from pathlib import Path

import pytest

from llm_conceptual_modeling.common.hf_transformers import DecodingConfig, RuntimeProfile
from llm_conceptual_modeling.hf_experiments import HFRunSpec


def _runtime_profile() -> RuntimeProfile:
    return RuntimeProfile(
        device="cuda",
        dtype="bfloat16",
        quantization="none",
        supports_thinking_toggle=False,
        context_limit=4096,
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
