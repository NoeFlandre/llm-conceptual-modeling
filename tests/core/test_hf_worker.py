from __future__ import annotations

import json
from pathlib import Path

from llm_conceptual_modeling.common.hf_transformers import DecodingConfig, RuntimeProfile
from llm_conceptual_modeling.hf_experiments import HFRunSpec
from llm_conceptual_modeling.hf_spec_codec import serialize_spec
from llm_conceptual_modeling.hf_worker import main


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
    assert worker_state["model_loaded"] is True
    assert worker_state["worker_pid"] > 0
    assert result_payload["ok"] is True
