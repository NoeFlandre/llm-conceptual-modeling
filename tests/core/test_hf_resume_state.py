import json
from pathlib import Path

from llm_conceptual_modeling.common.hf_transformers import DecodingConfig, RuntimeProfile
from llm_conceptual_modeling.hf_batch_types import HFRunSpec
from llm_conceptual_modeling.hf_resume_state import (
    build_seeded_resume_snapshot,
    load_valid_finished_summary,
    order_planned_specs_for_resume,
)


def _runtime_profile() -> RuntimeProfile:
    return RuntimeProfile(
        supports_thinking_toggle=False,
        quantization="none",
        device="cuda",
        dtype="bfloat16",
        context_limit=None,
    )


def _make_spec(
    *,
    pair_name: str,
    condition_bits: str,
    replication: int = 0,
    context_policy: dict[str, object] | None = None,
) -> HFRunSpec:
    return HFRunSpec(
        algorithm="algo1",
        model="allenai/Olmo-3-7B-Instruct",
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        pair_name=pair_name,
        condition_bits=condition_bits,
        condition_label="greedy",
        replication=replication,
        prompt_factors={},
        prompt_bundle=None,
        decoding=DecodingConfig(algorithm="greedy", temperature=0.0),
        input_payload={"graph": [], "subgraph1": [], "subgraph2": []},
        raw_context={"pair_name": pair_name, "Repetition": replication},
        seed=1,
        runtime_profile=_runtime_profile(),
        max_new_tokens_by_schema={"edge_list": 32, "vote_list": 16, "label_list": 16},
        context_policy=context_policy or {"resume_pass_mode": "throughput"},
    )


def _run_dir(output_root: Path, spec: HFRunSpec) -> Path:
    return (
        output_root
        / "runs"
        / spec.algorithm
        / spec.model.replace("/", "__")
        / spec.condition_label
        / spec.pair_name
        / spec.condition_bits
        / f"rep_{spec.replication:02d}"
    )


def test_load_valid_finished_summary_reclassifies_structurally_invalid_run(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (run_dir / "state.json").write_text('{"status":"finished"}', encoding="utf-8")
    (run_dir / "runtime.json").write_text("{}", encoding="utf-8")
    (run_dir / "raw_response.json").write_text("[]", encoding="utf-8")
    (run_dir / "raw_row.json").write_text(
        json.dumps({"Result": "[('1', '2')]"}),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"status": "finished"}),
        encoding="utf-8",
    )

    def validator(*, algorithm: str, raw_row: dict[str, object]) -> None:
        _ = algorithm
        _ = raw_row
        raise ValueError("Structurally invalid algo1 result: non-textual edge endpoint ('1', '2').")

    actual = load_valid_finished_summary(
        run_dir=run_dir,
        algorithm="algo1",
        validate_structural_runtime_result_fn=validator,
    )

    assert actual is None
    assert json.loads((run_dir / "state.json").read_text(encoding="utf-8"))["status"] == "failed"
    assert (run_dir / "error.json").exists()
    assert not (run_dir / "summary.json").exists()


def test_build_seeded_resume_snapshot_counts_finished_and_deferred_timeout(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    finished_spec = _make_spec(pair_name="sg1_sg2", condition_bits="00000")
    failed_spec = _make_spec(pair_name="sg1_sg2", condition_bits="00001")
    pending_spec = _make_spec(pair_name="sg1_sg2", condition_bits="00010")
    planned_specs = [finished_spec, failed_spec, pending_spec]

    finished_dir = _run_dir(output_root, finished_spec)
    finished_dir.mkdir(parents=True, exist_ok=True)
    (finished_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (finished_dir / "state.json").write_text('{"status":"finished"}', encoding="utf-8")
    (finished_dir / "runtime.json").write_text("{}", encoding="utf-8")
    (finished_dir / "raw_response.json").write_text("[]", encoding="utf-8")
    (finished_dir / "raw_row.json").write_text(
        json.dumps({"Result": "[('alpha', 'gamma')]"}),
        encoding="utf-8",
    )
    (finished_dir / "summary.json").write_text(
        json.dumps({"status": "finished", "pair_name": "sg1_sg2"}),
        encoding="utf-8",
    )

    failed_dir = _run_dir(output_root, failed_spec)
    failed_dir.mkdir(parents=True, exist_ok=True)
    (failed_dir / "state.json").write_text('{"status":"failed"}', encoding="utf-8")
    (failed_dir / "error.json").write_text(
        json.dumps(
            {
                "type": "MonitoredCommandTimeout",
                "message": "Monitored command exceeded stage timeout of 20.0 seconds.",
            }
        ),
        encoding="utf-8",
    )

    (output_root / "batch_status.json").parent.mkdir(parents=True, exist_ok=True)
    (output_root / "batch_status.json").write_text(
        json.dumps({"started_at": "2026-04-01T00:00:00+00:00"}),
        encoding="utf-8",
    )

    snapshot, summary_rows, seeded_finished, seeded_failed = build_seeded_resume_snapshot(
        output_root=output_root,
        planned_specs=planned_specs,
        started_at="2026-04-01T00:00:01+00:00",
        run_dir_for_spec_fn=_run_dir,
        current_run_payload_fn=lambda **payload: payload,
        status_timestamp_now_fn=lambda: "2026-04-01T00:00:02+00:00",
        validate_structural_runtime_result_fn=lambda **_: None,
    )

    assert snapshot["finished_count"] == 1
    assert snapshot["failed_count"] == 1
    assert snapshot["pending_count"] == 1
    assert len(summary_rows) == 1
    assert seeded_finished == {finished_dir}
    assert seeded_failed == {failed_dir}


def test_order_planned_specs_for_resume_prioritizes_low_timeout_risk_pairs(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    specs = [
        _make_spec(pair_name="sg2_sg3", condition_bits="00000"),
        _make_spec(pair_name="sg3_sg1", condition_bits="00000"),
        _make_spec(pair_name="sg1_sg2", condition_bits="00000"),
    ]

    hot_run_dir = (
        output_root
        / "runs"
        / "algo1"
        / "allenai__Olmo-3-7B-Instruct"
        / "greedy"
        / "sg2_sg3"
        / "11111"
        / "rep_00"
    )
    hot_run_dir.mkdir(parents=True, exist_ok=True)
    (hot_run_dir / "state.json").write_text('{"status":"failed"}', encoding="utf-8")
    (hot_run_dir / "error.json").write_text(
        json.dumps(
            {
                "type": "MonitoredCommandTimeout",
                "message": "Monitored command exceeded stage timeout of 20.0 seconds.",
            }
        ),
        encoding="utf-8",
    )

    ordered = order_planned_specs_for_resume(
        planned_specs=specs,
        output_root=output_root,
        resume=True,
        run_dir_for_spec_fn=_run_dir,
    )

    assert [spec.pair_name for spec in ordered] == ["sg1_sg2", "sg3_sg1", "sg2_sg3"]
