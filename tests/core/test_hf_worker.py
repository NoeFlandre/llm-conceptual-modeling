from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_conceptual_modeling.common.hf_transformers import DecodingConfig, RuntimeProfile
from llm_conceptual_modeling.hf_experiments import HFRunSpec
from llm_conceptual_modeling.hf_spec_codec import deserialize_spec, serialize_spec
from llm_conceptual_modeling.hf_worker import main, serve_request_queue
from llm_conceptual_modeling.hf_worker_request import enqueue_worker_request, load_worker_request


def test_hf_worker_writes_worker_state_before_running(monkeypatch, tmp_path: Path) -> None:
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
        runtime_profile=RuntimeProfile(
            device="cuda",
            dtype="bfloat16",
            quantization="none",
            supports_thinking_toggle=False,
            context_limit=4096,
        ),
    )
    spec_json = tmp_path / "spec.json"
    result_json = tmp_path / "result.json"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    spec_json.write_text(json.dumps(serialize_spec(spec)), encoding="utf-8")

    monkeypatch.setattr(
        "llm_conceptual_modeling.hf_worker.build_runtime_factory",
        lambda hf_token=None: object(),
    )
    monkeypatch.setattr(
        "llm_conceptual_modeling.hf_experiments._run_algo1",
        lambda spec, *, hf_runtime, run_dir=None: {
            "raw_row": {
                "pair_name": "sg1_sg2",
                "Repetition": 0,
                "Result": "[('alpha', 'gamma')]",
                "graph": "[('alpha', 'gamma')]",
                "subgraph1": "[('alpha', 'beta')]",
                "subgraph2": "[('gamma', 'delta')]",
            },
            "runtime": {"thinking_mode_supported": False},
            "raw_response": "{}",
        },
    )

    exit_code = main(
        [
            "--spec-json",
            str(spec_json),
            "--result-json",
            str(result_json),
            "--run-dir",
            str(run_dir),
        ]
    )

    worker_state = json.loads((run_dir / "worker_state.json").read_text(encoding="utf-8"))
    result_payload = json.loads(result_json.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert worker_state["model_loaded"] is False
    assert worker_state["phase"] == "prefetching_model"
    assert worker_state["worker_pid"] > 0
    assert result_payload["ok"] is True


def test_persistent_hf_worker_serves_two_requests_with_one_model_load(
    monkeypatch,
    tmp_path: Path,
) -> None:
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    build_calls = {"count": 0}
    cache_release_calls = {"count": 0}
    served_pairs: list[str] = []

    def make_spec(pair_name: str) -> HFRunSpec:
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
        )

    def queue_request(index: int, pair_name: str) -> tuple[Path, Path]:
        run_dir = tmp_path / f"run_{index}"
        run_dir.mkdir()
        spec_json = run_dir / "spec.json"
        result_json = run_dir / "result.json"
        spec_json.write_text(json.dumps(serialize_spec(make_spec(pair_name))), encoding="utf-8")
        request_json = queue_dir / f"{index:04d}.request.json"
        request_json.write_text(
            json.dumps(
                {
                    "spec_json": str(spec_json),
                    "result_json": str(result_json),
                    "run_dir": str(run_dir),
                }
            ),
            encoding="utf-8",
        )
        return run_dir, result_json

    first_run_dir, first_result_json = queue_request(1, "sg1_sg2")
    second_run_dir, second_result_json = queue_request(2, "sg2_sg3")

    def fake_build_runtime_factory(hf_token=None):
        _ = hf_token
        build_calls["count"] += 1
        return object()

    def fake_run_algo1(spec, *, hf_runtime, run_dir=None):
        _ = hf_runtime
        served_pairs.append(spec.pair_name)
        return {
            "raw_row": {
                "pair_name": spec.pair_name,
                "Repetition": 0,
                "Result": "[('alpha', 'gamma')]",
                "graph": "[('alpha', 'gamma')]",
                "subgraph1": "[('alpha', 'beta')]",
                "subgraph2": "[('gamma', 'delta')]",
            },
            "runtime": {"thinking_mode_supported": False},
            "raw_response": "{}",
        }

    monkeypatch.setattr(
        "llm_conceptual_modeling.hf_worker.build_runtime_factory",
        fake_build_runtime_factory,
    )
    monkeypatch.setattr(
        "llm_conceptual_modeling.hf_experiments._run_algo1",
        fake_run_algo1,
    )
    monkeypatch.setattr(
        "llm_conceptual_modeling.hf_worker._release_runtime_cache",
        lambda: cache_release_calls.__setitem__("count", cache_release_calls["count"] + 1),
    )

    served_count = serve_request_queue(queue_dir=queue_dir, max_requests=2, idle_sleep_seconds=0.0)

    first_worker_state = json.loads(
        (first_run_dir / "worker_state.json").read_text(encoding="utf-8")
    )
    second_worker_state = json.loads(
        (second_run_dir / "worker_state.json").read_text(encoding="utf-8")
    )
    assert served_count == 2
    assert build_calls["count"] == 1
    assert served_pairs == ["sg1_sg2", "sg2_sg3"]
    assert json.loads(first_result_json.read_text(encoding="utf-8"))["ok"] is True
    assert json.loads(second_result_json.read_text(encoding="utf-8"))["ok"] is True
    assert first_worker_state["model_loaded"] is False
    assert second_worker_state["model_loaded"] is False
    assert first_worker_state["phase"] == "prefetching_model"
    assert second_worker_state["phase"] == "prefetching_model"
    assert first_worker_state["requests_served_by_process"] == 1
    assert second_worker_state["requests_served_by_process"] == 2
    assert cache_release_calls["count"] == 2


def test_persistent_hf_worker_skips_stale_finished_requests(
    monkeypatch,
    tmp_path: Path,
) -> None:
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    build_calls = {"count": 0}
    executed_runs: list[str] = []

    def make_spec(pair_name: str) -> HFRunSpec:
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
        )

    def write_finished_run(run_dir: Path, result_json: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
        (run_dir / "runtime.json").write_text("{}", encoding="utf-8")
        (run_dir / "raw_response.json").write_text("{}", encoding="utf-8")
        (run_dir / "raw_row.json").write_text("{}", encoding="utf-8")
        (run_dir / "state.json").write_text(json.dumps({"status": "finished"}), encoding="utf-8")
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "status": "finished",
                    "accuracy": 1.0,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                }
            ),
            encoding="utf-8",
        )
        result_json.write_text(
            json.dumps({"ok": True, "runtime_result": {"raw_row": {"pair_name": "sg1_sg2"}}}),
            encoding="utf-8",
        )

    stale_run_dir = tmp_path / "finished_run"
    stale_result_json = stale_run_dir / "worker_result.json"
    write_finished_run(stale_run_dir, stale_result_json)
    queue_request = queue_dir / "0001.request.json"
    queue_request.write_text(
        json.dumps(
            {
                "spec_json": str(stale_run_dir / "spec.json"),
                "result_json": str(stale_result_json),
                "run_dir": str(stale_run_dir),
            }
        ),
        encoding="utf-8",
    )

    active_run_dir = tmp_path / "active_run"
    active_run_dir.mkdir()
    active_result_json = active_run_dir / "worker_result.json"
    active_spec = make_spec("sg2_sg3")
    (active_run_dir / "spec.json").write_text(
        json.dumps(serialize_spec(active_spec)),
        encoding="utf-8",
    )
    active_request = queue_dir / "0002.request.json"
    active_request.write_text(
        json.dumps(
            {
                "spec_json": str(active_run_dir / "spec.json"),
                "result_json": str(active_result_json),
                "run_dir": str(active_run_dir),
            }
        ),
        encoding="utf-8",
    )

    def fake_build_runtime_factory(hf_token=None):
        _ = hf_token
        build_calls["count"] += 1
        return object()

    def fake_execute_request(*, spec_json_path, result_json_path, run_dir, hf_runtime, requests_served_by_process=1):
        _ = spec_json_path, hf_runtime, requests_served_by_process
        executed_runs.append(run_dir.name)
        result_json_path.write_text(
            json.dumps({"ok": True, "runtime_result": {"raw_row": {"pair_name": run_dir.name}}}),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(
        "llm_conceptual_modeling.hf_worker.build_runtime_factory",
        fake_build_runtime_factory,
    )
    monkeypatch.setattr(
        "llm_conceptual_modeling.hf_worker._execute_request",
        fake_execute_request,
    )
    monkeypatch.setattr(
        "llm_conceptual_modeling.hf_worker._release_runtime_cache",
        lambda: None,
    )

    served_count = serve_request_queue(queue_dir=queue_dir, max_requests=2, idle_sleep_seconds=0.0)

    assert served_count == 1
    assert build_calls["count"] == 1
    assert executed_runs == ["active_run"]
    assert not queue_request.exists()
    assert not active_request.exists()
    assert stale_result_json.exists()
    assert active_result_json.exists()


def test_persistent_hf_worker_recovers_claimed_unfinished_requests(
    monkeypatch,
    tmp_path: Path,
) -> None:
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    build_calls = {"count": 0}
    executed_runs: list[str] = []

    run_dir = tmp_path / "unfinished_run"
    run_dir.mkdir()
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
        runtime_profile=RuntimeProfile(
            device="cuda",
            dtype="bfloat16",
            quantization="none",
            supports_thinking_toggle=False,
            context_limit=4096,
        ),
    )
    spec_json = run_dir / "spec.json"
    result_json = run_dir / "worker_result.json"
    spec_json.write_text(json.dumps(serialize_spec(spec)), encoding="utf-8")
    claimed_request = queue_dir / "0001.claimed.json"
    claimed_request.write_text(
        json.dumps(
            {
                "spec_json": str(spec_json),
                "result_json": str(result_json),
                "run_dir": str(run_dir),
            }
        ),
        encoding="utf-8",
    )

    def fake_build_runtime_factory(hf_token=None):
        _ = hf_token
        build_calls["count"] += 1
        return object()

    def fake_execute_request(*, spec_json_path, result_json_path, run_dir, hf_runtime, requests_served_by_process=1):
        _ = spec_json_path, hf_runtime, requests_served_by_process
        executed_runs.append(run_dir.name)
        result_json_path.write_text(
            json.dumps({"ok": True, "runtime_result": {"raw_row": {"pair_name": run_dir.name}}}),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(
        "llm_conceptual_modeling.hf_worker.build_runtime_factory",
        fake_build_runtime_factory,
    )
    monkeypatch.setattr(
        "llm_conceptual_modeling.hf_worker._execute_request",
        fake_execute_request,
    )
    monkeypatch.setattr(
        "llm_conceptual_modeling.hf_worker._release_runtime_cache",
        lambda: None,
    )

    served_count = serve_request_queue(queue_dir=queue_dir, max_requests=1, idle_sleep_seconds=0.0)

    assert served_count == 1
    assert build_calls["count"] == 1
    assert executed_runs == ["unfinished_run"]
    assert not claimed_request.exists()
    assert result_json.exists()


def test_worker_request_round_trips_shared_queue_artifacts(tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
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
        runtime_profile=RuntimeProfile(
            device="cuda",
            dtype="bfloat16",
            quantization="none",
            supports_thinking_toggle=False,
            context_limit=4096,
        ),
    )

    enqueued = enqueue_worker_request(
        queue_dir=queue_dir,
        run_dir=run_dir,
        spec=spec,
        request_id="00000042",
    )
    loaded = load_worker_request(enqueued.request_json_path)

    assert loaded.request_id == "00000042"
    assert loaded.request_json_path == queue_dir / "00000042.request.json"
    assert loaded.spec_json_path == run_dir / "worker_spec.json"
    assert loaded.result_json_path == run_dir / "worker_result.json"
    assert loaded.run_dir == run_dir
    spec_payload = json.loads(loaded.spec_json_path.read_text(encoding="utf-8"))

    assert serialize_spec(deserialize_spec(spec_payload)) == spec_payload


def test_deserialize_spec_ignores_malformed_seed_values() -> None:
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
        runtime_profile=RuntimeProfile(
            device="cuda",
            dtype="bfloat16",
            quantization="none",
            supports_thinking_toggle=False,
            context_limit=4096,
        ),
        base_seed=11,
        seed=13,
    )
    payload = serialize_spec(spec)
    payload["base_seed"] = "false"
    payload["seed"] = "false"

    decoded = deserialize_spec(payload)

    assert decoded.base_seed == 0
    assert decoded.seed == 0


def test_deserialize_spec_rejects_boolean_replication_value() -> None:
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
        runtime_profile=RuntimeProfile(
            device="cuda",
            dtype="bfloat16",
            quantization="none",
            supports_thinking_toggle=False,
            context_limit=4096,
        ),
    )
    payload = serialize_spec(spec)
    payload["replication"] = True

    with pytest.raises(TypeError, match="Spec field replication must be an integer, got bool"):
        deserialize_spec(payload)
