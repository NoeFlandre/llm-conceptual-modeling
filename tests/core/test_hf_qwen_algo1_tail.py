from __future__ import annotations

import json
from pathlib import Path

import yaml
import pytest

from llm_conceptual_modeling.hf_qwen_algo1_tail import (
    QWEN_ALGO1_TAIL_CONDITION_LABEL,
    QWEN_ALGO1_TAIL_EXPECTED_COUNT,
    QWEN_ALGO1_TAIL_MODEL,
    build_qwen_algo1_tail_preflight_report,
    collect_qwen_algo1_tail_records,
    prepare_qwen_algo1_tail_bundle,
)


def _write_canonical_runtime_config(path: Path) -> None:
    path.write_text(
        """
run:
  provider: hf-transformers
  output_root: /workspace/results/hf-paper-batch-canonical
  replications: 5
runtime:
  seed: 20260331
  temperature: 0.0
  quantization: none
  device_policy: cuda-only
  context_policy:
    prompt_truncation: forbid
    safety_margin_tokens: 64
    generation_timeout_seconds: 45.0
    retry_timeout_failures_on_resume: true
    retry_structural_failures_on_resume: true
    worker_process_mode: persistent
    max_requests_per_worker_process: 16
  max_new_tokens_by_schema:
    edge_list: 256
    vote_list: 64
    label_list: 128
    children_by_label: 384
  thinking_mode_by_model:
    mistralai/Ministral-3-8B-Instruct-2512: acknowledged-unsupported
    Qwen/Qwen3.5-9B: disabled
models:
  chat_models:
  - mistralai/Ministral-3-8B-Instruct-2512
  - Qwen/Qwen3.5-9B
  embedding_model: Qwen/Qwen3-Embedding-0.6B
decoding:
- algorithm: greedy
  num_beams: null
  penalty_alpha: null
  top_k: null
  temperature: 0.0
- algorithm: contrastive
  num_beams: null
  penalty_alpha: 0.8
  top_k: 4
  temperature: 0.0
algorithms:
  algo1:
    base_fragments: []
    factors: {}
    fragment_definitions: {}
    prompt_templates:
      body: "Task."
      direct_edge: ""
    pair_names:
    - sg1_sg2
    - sg2_sg3
  algo2:
    base_fragments: []
    factors: {}
    fragment_definitions: {}
    prompt_templates:
      body: "Task."
inputs:
  graph_source: default
shared_fragments: {}
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _write_tail_ledger(path: Path) -> None:
    records = []
    for bits in ("00101", "10100"):
        for replication in range(5):
            records.append(
                {
                    "identity": {
                        "algorithm": "algo1",
                        "condition_bits": bits,
                        "condition_label": QWEN_ALGO1_TAIL_CONDITION_LABEL,
                        "model": QWEN_ALGO1_TAIL_MODEL,
                        "pair_name": "sg1_sg2",
                        "replication": replication,
                    },
                    "status": "retryable_failed",
                }
            )
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-12T00:00:00+00:00",
                "expected_total_runs": len(records),
                "finished_count": 0,
                "pending_count": 0,
                "retryable_failed_count": len(records),
                "terminal_failed_count": 0,
                "records": records,
            }
        ),
        encoding="utf-8",
    )


def test_collect_qwen_algo1_tail_records_requires_exact_expected_surface(tmp_path: Path) -> None:
    results_root = tmp_path / "canonical"
    results_root.mkdir()
    _write_tail_ledger(results_root / "ledger.json")

    records = collect_qwen_algo1_tail_records(results_root)

    assert len(records) == QWEN_ALGO1_TAIL_EXPECTED_COUNT
    assert {record["identity"]["condition_bits"] for record in records} == {"00101", "10100"}
    assert {record["identity"]["pair_name"] for record in records} == {"sg1_sg2"}


def test_collect_qwen_algo1_tail_records_rejects_broadened_surface(tmp_path: Path) -> None:
    results_root = tmp_path / "canonical"
    results_root.mkdir()
    _write_tail_ledger(results_root / "ledger.json")
    payload = json.loads((results_root / "ledger.json").read_text(encoding="utf-8"))
    payload["records"].append(
        {
            "identity": {
                "algorithm": "algo1",
                "condition_bits": "11111",
                "condition_label": QWEN_ALGO1_TAIL_CONDITION_LABEL,
                "model": QWEN_ALGO1_TAIL_MODEL,
                "pair_name": "sg1_sg2",
                "replication": 0,
            },
            "status": "retryable_failed",
        }
    )
    (results_root / "ledger.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown condition_bits"):
        collect_qwen_algo1_tail_records(results_root)


def test_collect_qwen_algo1_tail_records_ignores_finished_history_outside_the_tail(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "canonical"
    results_root.mkdir()
    _write_tail_ledger(results_root / "ledger.json")
    payload = json.loads((results_root / "ledger.json").read_text(encoding="utf-8"))
    payload["records"].append(
        {
            "identity": {
                "algorithm": "algo1",
                "condition_bits": "11111",
                "condition_label": QWEN_ALGO1_TAIL_CONDITION_LABEL,
                "model": QWEN_ALGO1_TAIL_MODEL,
                "pair_name": "sg2_sg3",
                "replication": 4,
            },
            "status": "finished",
        }
    )
    (results_root / "ledger.json").write_text(json.dumps(payload), encoding="utf-8")

    records = collect_qwen_algo1_tail_records(results_root)

    assert len(records) == QWEN_ALGO1_TAIL_EXPECTED_COUNT


def test_prepare_qwen_algo1_tail_bundle_writes_restricted_manifest_and_config(tmp_path: Path) -> None:
    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    _write_tail_ledger(canonical_root / "ledger.json")
    _write_canonical_runtime_config(canonical_root / "runtime_config.yaml")
    for bits in ("00101", "10100"):
        for replication in range(5):
            run_dir = (
                canonical_root
                / "runs"
                / "algo1"
                / "Qwen__Qwen3.5-9B"
                / "contrastive_penalty_alpha_0.8"
                / "sg1_sg2"
                / bits
                / f"rep_{replication:02d}"
            )
            run_dir.mkdir(parents=True)
            (run_dir / "error.json").write_text('{"type":"RuntimeError","message":"bad"}', encoding="utf-8")
            (run_dir / "state.json").write_text('{"status":"failed"}', encoding="utf-8")

    tail_parent = tmp_path / "tail-parent"
    tail_root = tail_parent / "hf-paper-batch-qwen-algo1-tail"

    report = prepare_qwen_algo1_tail_bundle(
        canonical_results_root=canonical_root,
        tail_results_root=tail_root,
        remote_output_root="/workspace/results/qwen-tail/hf-paper-batch-qwen-algo1-tail",
    )

    manifest = json.loads((tail_root / "shard_manifest.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((tail_root / "runtime_config.yaml").read_text(encoding="utf-8"))
    seed_ledger = json.loads((tail_root / "ledger.json").read_text(encoding="utf-8"))

    assert report["identity_count"] == 10
    assert len(manifest["identities"]) == 10
    assert manifest["active_chat_models"] == [QWEN_ALGO1_TAIL_MODEL]
    assert config["models"]["chat_models"] == [QWEN_ALGO1_TAIL_MODEL]
    assert list(config["algorithms"]) == ["algo1"]
    assert config["algorithms"]["algo1"]["pair_names"] == ["sg1_sg2"]
    assert [item["algorithm"] for item in config["decoding"]] == ["contrastive"]
    assert [item["penalty_alpha"] for item in config["decoding"]] == [0.8]
    assert config["runtime"]["context_policy"]["worker_process_mode"] == "persistent"
    assert config["runtime"]["context_policy"]["retry_structural_failures_on_resume"] is True
    assert config["runtime"]["context_policy"]["max_requests_per_worker_process"] == 64
    assert config["runtime"]["max_new_tokens_by_schema"]["edge_list"] == 512
    assert config["run"]["output_root"] == "/workspace/results/qwen-tail/hf-paper-batch-qwen-algo1-tail"
    assert len(seed_ledger["records"]) == 10
    assert all(record["status"] == "retryable_failed" for record in seed_ledger["records"])
    assert len(report["copied_run_dirs"]) == 10


def test_build_qwen_algo1_tail_preflight_report_rejects_degraded_watcher(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "data" / "inputs").mkdir(parents=True)
    canonical_parent = tmp_path / "canonical-parent"
    canonical_root = canonical_parent / "hf-paper-batch-canonical"
    canonical_root.mkdir(parents=True)
    _write_tail_ledger(canonical_root / "ledger.json")
    _write_canonical_runtime_config(canonical_root / "runtime_config.yaml")

    tail_parent = tmp_path / "tail-parent"
    tail_root = tail_parent / "hf-paper-batch-qwen-algo1-tail"
    prepare_qwen_algo1_tail_bundle(
        canonical_results_root=canonical_root,
        tail_results_root=tail_root,
        remote_output_root="/workspace/results/qwen-tail/hf-paper-batch-qwen-algo1-tail",
    )
    (tail_root / "results-sync-status.json").write_text(
        json.dumps({"status": "degraded"}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="watcher is degraded"):
        build_qwen_algo1_tail_preflight_report(
            repo_root=repo_root,
            canonical_results_root=canonical_root,
            tail_results_root=tail_root,
        )
