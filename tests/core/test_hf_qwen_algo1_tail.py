from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints

import pytest
import yaml

from llm_conceptual_modeling.common.hf_transformers._policy import DecodingConfig
from llm_conceptual_modeling.hf_tail import qwen_algo1 as tail
from llm_conceptual_modeling.hf_tail.qwen_algo1 import (
    QWEN_ALGO1_TAIL_CONDITION_LABEL,
    QWEN_ALGO1_TAIL_EXPECTED_COUNT,
    QWEN_ALGO1_TAIL_MODEL,
    QwenAlgo1TailPreflightReport,
    build_qwen_algo1_tail_preflight_report,
    collect_qwen_algo1_tail_records,
    prepare_qwen_algo1_tail_bundle,
    write_qwen_algo1_tail_config,
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


def _seed_tail_run_dirs(root: Path) -> None:
    for bits in ("00101", "10100"):
        for replication in range(5):
            run_dir = (
                root
                / "runs"
                / "algo1"
                / "Qwen__Qwen3.5-9B"
                / "contrastive_penalty_alpha_0.8"
                / "sg1_sg2"
                / bits
                / f"rep_{replication:02d}"
            )
            run_dir.mkdir(parents=True)
            (run_dir / "error.json").write_text(
                '{"type":"RuntimeError","message":"bad"}',
                encoding="utf-8",
            )
            (run_dir / "state.json").write_text('{"status":"failed"}', encoding="utf-8")


def test_collect_qwen_algo1_tail_records_requires_exact_expected_surface(tmp_path: Path) -> None:
    results_root = tmp_path / "canonical"
    results_root.mkdir()
    _write_tail_ledger(results_root / "ledger.json")

    records = collect_qwen_algo1_tail_records(results_root)

    assert len(records) == QWEN_ALGO1_TAIL_EXPECTED_COUNT
    assert {record["identity"]["condition_bits"] for record in records} == {"00101", "10100"}
    assert {record["identity"]["pair_name"] for record in records} == {"sg1_sg2"}
    assert [
        (record["identity"]["condition_bits"], record["identity"]["replication"])
        for record in records
    ] == [(bits, replication) for bits in ("00101", "10100") for replication in range(5)]
    assert {record["status"] for record in records} == {"retryable_failed"}


def test_collect_qwen_algo1_tail_records_uses_canonical_ledger_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_root = tmp_path / "canonical"
    results_root.mkdir()
    observed_paths: list[Path] = []
    records = [
        _tail_record(condition_bits=bits, replication=replication)
        for bits in ("00101", "10100")
        for replication in range(5)
    ]

    def read_tail_seed_ledger(path: Path) -> dict[str, object]:
        observed_paths.append(path)
        return {"records": records}

    monkeypatch.setattr(tail, "_read_tail_seed_ledger", read_tail_seed_ledger)

    assert collect_qwen_algo1_tail_records(results_root) == records
    assert observed_paths == [results_root.resolve() / "ledger.json"]


def test_collect_qwen_algo1_tail_records_rejects_non_list_records_payload(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "canonical"
    results_root.mkdir()
    (results_root / "ledger.json").write_text(
        json.dumps({"records": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        collect_qwen_algo1_tail_records(results_root)

    assert str(exc_info.value) == "ledger.json does not contain a records list."


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


def test_prepare_qwen_algo1_tail_bundle_writes_restricted_manifest_and_config(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    _write_tail_ledger(canonical_root / "ledger.json")
    _write_canonical_runtime_config(canonical_root / "runtime_config.yaml")
    _seed_tail_run_dirs(canonical_root)

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
    assert report["canonical_results_root"] == str(canonical_root.resolve())
    assert report["tail_results_root"] == str(tail_root.resolve())
    assert report["remote_output_root"] == (
        "/workspace/results/qwen-tail/hf-paper-batch-qwen-algo1-tail"
    )
    assert report["manifest_path"] == str(tail_root / "shard_manifest.json")
    assert report["config_path"] == str(tail_root / "runtime_config.yaml")
    assert report["ledger_path"] == str(tail_root / "ledger.json")
    expected_copied_run_dirs = [
        str(
            tail_root
            / "runs"
            / "algo1"
            / "Qwen__Qwen3.5-9B"
            / QWEN_ALGO1_TAIL_CONDITION_LABEL
            / "sg1_sg2"
            / bits
            / f"rep_{replication:02d}"
        )
        for bits in ("00101", "10100")
        for replication in range(5)
    ]
    assert report["copied_run_dirs"] == expected_copied_run_dirs
    assert len(manifest["identities"]) == 10
    assert manifest["active_chat_models"] == [QWEN_ALGO1_TAIL_MODEL]
    assert config["models"]["chat_models"] == [QWEN_ALGO1_TAIL_MODEL]
    assert list(config["algorithms"]) == ["algo1"]
    assert config["algorithms"]["algo1"]["pair_names"] == ["sg1_sg2", "sg2_sg3"]
    assert [item["algorithm"] for item in config["decoding"]] == ["contrastive"]
    assert [item["penalty_alpha"] for item in config["decoding"]] == [0.8]
    assert config["runtime"]["context_policy"]["worker_process_mode"] == "persistent"
    assert config["runtime"]["context_policy"]["retry_structural_failures_on_resume"] is True
    assert config["runtime"]["context_policy"]["max_requests_per_worker_process"] == 64
    assert config["runtime"]["max_new_tokens_by_schema"]["edge_list"] == 512
    assert config["run"]["replications"] == 5
    assert (
        config["run"]["output_root"]
        == "/workspace/results/qwen-tail/hf-paper-batch-qwen-algo1-tail"
    )
    assert len(seed_ledger["records"]) == 10
    assert all(record["status"] == "retryable_failed" for record in seed_ledger["records"])
    assert len(report["copied_run_dirs"]) == 10
    assert report["copied_run_dir_count"] == 10
    assert report["seed_status_counts"] == {"retryable_failed": 10}
    assert (tail_root / "runtime_config.yaml").read_text(encoding="utf-8") == yaml.safe_dump(
        config,
        sort_keys=False,
    )

    second_report = prepare_qwen_algo1_tail_bundle(
        canonical_results_root=canonical_root,
        tail_results_root=tail_root,
        remote_output_root="/workspace/results/qwen-tail/hf-paper-batch-qwen-algo1-tail",
    )
    assert second_report["copied_run_dir_count"] == 10
    assert second_report["copied_run_dirs"] == expected_copied_run_dirs


def test_write_qwen_algo1_tail_config_rejects_missing_contrastive_decoding(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    _write_canonical_runtime_config(canonical_root / "runtime_config.yaml")
    config = yaml.safe_load(
        (canonical_root / "runtime_config.yaml").read_text(encoding="utf-8")
    )
    config["decoding"] = config["decoding"][:1]
    (canonical_root / "runtime_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError) as exc_info:
        write_qwen_algo1_tail_config(
            canonical_results_root=canonical_root,
            tail_results_root=tmp_path / "tail",
            remote_output_root="/workspace/results/qwen-tail",
        )
    assert str(exc_info.value) == (
        "Canonical config does not contain the expected contrastive decoding entry."
    )


def test_build_tail_runtime_config_requires_algorithm_and_penalty_pair(tmp_path: Path) -> None:
    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    _write_canonical_runtime_config(canonical_root / "runtime_config.yaml")
    canonical_payload = yaml.safe_load(
        (canonical_root / "runtime_config.yaml").read_text(encoding="utf-8")
    )
    canonical_payload["run"]["replications"] = 3
    canonical_payload["runtime"]["context_policy"]["worker_process_mode"] = "subprocess"
    canonical_payload["runtime"]["context_policy"][
        "retry_structural_failures_on_resume"
    ] = False
    (canonical_root / "runtime_config.yaml").write_text(
        yaml.safe_dump(canonical_payload, sort_keys=False),
        encoding="utf-8",
    )
    canonical_config = tail.load_hf_run_config(canonical_root / "runtime_config.yaml")
    canonical_config = replace(
        canonical_config,
        decoding=(
            *canonical_config.decoding,
            DecodingConfig(algorithm="greedy", penalty_alpha=0.8),
        ),
    )

    payload = tail._build_tail_runtime_config(
        canonical_config=canonical_config,
        remote_output_root="/workspace/results/qwen-tail",
    )

    assert payload["decoding"] == [
        {
            "algorithm": "contrastive",
            "num_beams": None,
            "penalty_alpha": 0.8,
            "top_k": 4,
            "temperature": 0.0,
        }
    ]
    assert payload["run"]["output_root"] == "/workspace/results/qwen-tail"
    assert payload["run"]["replications"] == 5
    assert payload["models"]["chat_models"] == [QWEN_ALGO1_TAIL_MODEL]
    assert payload["runtime"]["context_policy"]["worker_process_mode"] == "persistent"
    assert payload["runtime"]["context_policy"]["retry_structural_failures_on_resume"] is True
    assert payload["runtime"]["context_policy"]["max_requests_per_worker_process"] == 64
    assert payload["runtime"]["max_new_tokens_by_schema"]["edge_list"] == 512


def test_write_tail_config_uses_canonical_path_and_stable_key_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_root = tmp_path / "canonical"
    tail_root = tmp_path / "tail"
    canonical_root.mkdir()
    tail_root.mkdir()
    _write_canonical_runtime_config(canonical_root / "runtime_config.yaml")
    observed_load_paths: list[Path] = []
    real_load_hf_run_config = tail.load_hf_run_config

    def load_hf_run_config(path: str | Path) -> object:
        observed_load_paths.append(Path(path))
        return real_load_hf_run_config(path)

    monkeypatch.setattr(tail, "load_hf_run_config", load_hf_run_config)
    safe_dump_calls: list[dict[str, object]] = []
    real_safe_dump = tail.yaml.safe_dump

    def safe_dump(data: object, *args: object, **kwargs: object) -> str:
        safe_dump_calls.append(dict(kwargs))
        return real_safe_dump(data, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(tail.yaml, "safe_dump", safe_dump)
    write_calls: list[tuple[Path, dict[str, object]]] = []
    real_write_text = Path.write_text

    def write_text(
        path: Path,
        data: str,
        *args: object,
        **kwargs: object,
    ) -> int:
        write_calls.append((path, dict(kwargs)))
        return real_write_text(path, data, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "write_text", write_text)

    config = tail.write_qwen_algo1_tail_config(
        canonical_results_root=canonical_root,
        tail_results_root=tail_root,
        remote_output_root="/workspace/results/qwen-tail",
    )

    assert observed_load_paths == [canonical_root / "runtime_config.yaml"]
    assert safe_dump_calls == [{"sort_keys": False}]
    assert write_calls == [(tail_root / "runtime_config.yaml", {"encoding": "utf-8"})]
    assert list(config) == [
        "run",
        "runtime",
        "models",
        "decoding",
        "inputs",
        "shared_fragments",
        "algorithms",
    ]
    assert (tail_root / "runtime_config.yaml").read_text(encoding="utf-8") == yaml.safe_dump(
        config,
        sort_keys=False,
    )


def test_prepare_qwen_algo1_tail_bundle_rejects_unscoped_tail_root(tmp_path: Path) -> None:
    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    _write_tail_ledger(canonical_root / "ledger.json")

    with pytest.raises(ValueError) as exc_info:
        prepare_qwen_algo1_tail_bundle(
            canonical_results_root=canonical_root,
            tail_results_root=tmp_path / "unscoped-tail",
            remote_output_root="/workspace/results/qwen-tail",
        )
    assert str(exc_info.value) == (
        "tail_results_root basename must start with 'hf-paper-batch-' so ledger refresh can "
        "discover it as an isolated batch root."
    )


def test_build_qwen_algo1_tail_preflight_report_rejects_degraded_watcher(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "data" / "inputs").mkdir(parents=True)
    canonical_parent = tmp_path / "canonical-parent"
    canonical_root = canonical_parent / "hf-paper-batch-canonical"
    canonical_root.mkdir(parents=True)
    _write_tail_ledger(canonical_root / "ledger.json")
    _write_canonical_runtime_config(canonical_root / "runtime_config.yaml")
    _seed_tail_run_dirs(canonical_root)

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

    with pytest.raises(RuntimeError) as exc_info:
        build_qwen_algo1_tail_preflight_report(
            repo_root=repo_root,
            canonical_results_root=canonical_root,
            tail_results_root=tail_root,
        )
    assert str(exc_info.value) == (
        "Dedicated tail watcher is degraded; refusing fresh-host prep."
    )


def test_build_qwen_algo1_tail_preflight_report_returns_resume_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "data" / "inputs").mkdir(parents=True)
    canonical_root = tmp_path / "canonical-parent" / "hf-paper-batch-canonical"
    canonical_root.mkdir(parents=True)
    _write_tail_ledger(canonical_root / "ledger.json")
    _write_canonical_runtime_config(canonical_root / "runtime_config.yaml")
    _seed_tail_run_dirs(canonical_root)

    tail_root = tmp_path / "tail-parent" / "hf-paper-batch-qwen-algo1-tail"
    prepare_qwen_algo1_tail_bundle(
        canonical_results_root=canonical_root,
        tail_results_root=tail_root,
        remote_output_root="/workspace/results/qwen-tail/hf-paper-batch-qwen-algo1-tail",
    )
    (tail_root / "results-sync-status.json").write_text(
        json.dumps({"status": "healthy"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "llm_conceptual_modeling.hf_tail.qwen_algo1.plan_paper_batch",
        lambda **_kwargs: [
            SimpleNamespace(
                algorithm="algo1",
                model=QWEN_ALGO1_TAIL_MODEL,
                condition_label=QWEN_ALGO1_TAIL_CONDITION_LABEL,
                pair_name="sg1_sg2",
                condition_bits=bits,
                replication=replication,
            )
            for bits in ("00101", "10100")
            for replication in range(5)
        ],
    )
    observed: dict[str, object] = {}

    real_read_watcher_status = tail.read_watcher_status

    def read_watcher_status(path: str | Path) -> dict[str, object]:
        observed["watcher_status_path"] = Path(path)
        return real_read_watcher_status(path)

    monkeypatch.setattr(tail, "read_watcher_status", read_watcher_status)

    def refresh_ledger(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {
            "expected_total_runs": 10,
            "finished_count": 4,
            "pending_count": 6,
            "retryable_failed_count": 4,
            "terminal_failed_count": 0,
        }

    monkeypatch.setattr(tail, "refresh_ledger", refresh_ledger)

    real_read_tail_manifest = tail._read_tail_manifest

    def read_tail_manifest(path: Path) -> tail.QwenAlgo1TailManifest:
        observed["manifest_path"] = path
        return real_read_tail_manifest(path)

    monkeypatch.setattr(tail, "_read_tail_manifest", read_tail_manifest)

    real_load_hf_run_config = tail.load_hf_run_config

    def load_hf_run_config(path: str | Path) -> object:
        observed["config_path"] = Path(path)
        return real_load_hf_run_config(path)

    monkeypatch.setattr(tail, "load_hf_run_config", load_hf_run_config)

    report = build_qwen_algo1_tail_preflight_report(
        repo_root=repo_root,
        canonical_results_root=canonical_root,
        tail_results_root=tail_root,
    )

    assert report["repo_root"] == str(repo_root.resolve())
    assert report["canonical_results_root"] == str(canonical_root.resolve())
    assert report["tail_results_root"] == str(tail_root.resolve())
    assert report["watcher_status"] == {"status": "healthy"}
    assert observed["watcher_status_path"] == tail_root / "results-sync-status.json"
    assert observed["results_root"] == tail_root.parent
    assert observed["ledger_root"] == tail_root
    assert observed["manifest_path"] == tail_root / "shard_manifest.json"
    assert observed["config_path"] == tail_root / "runtime_config.yaml"
    assert report["canonical_ledger"] == {
        "finished_count": 0,
        "pending_count": 0,
        "retryable_failed_count": 10,
        "terminal_failed_count": 0,
        "expected_total_runs": 10,
    }
    assert report["tail_ledger"] == {
        "finished_count": 4,
        "pending_count": 6,
        "retryable_failed_count": 4,
        "terminal_failed_count": 0,
        "expected_total_runs": 10,
    }
    assert report["tail_ledger"]["expected_total_runs"] == 10
    assert report["resume_preflight"] == {
        "results_root": str(tail_root.resolve()),
        "total_runs": 10,
        "finished_count": 4,
        "failed_count": 0,
        "pending_count": 10,
        "running_count": 0,
        "can_resume": True,
        "resume_mode": "resume",
    }

    manifest_path = tail_root / "shard_manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest_payload["identities"]
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    captured_manifest_identities: list[object] = []

    def capture_manifest_identities(identities: object) -> None:
        captured_manifest_identities.append(identities)
        raise RuntimeError("captured manifest identities")

    monkeypatch.setattr(tail, "_validate_tail_manifest_identities", capture_manifest_identities)
    with pytest.raises(RuntimeError, match="captured manifest identities"):
        build_qwen_algo1_tail_preflight_report(
            repo_root=repo_root,
            canonical_results_root=canonical_root,
            tail_results_root=tail_root,
        )
    assert captured_manifest_identities == [[]]


def test_hf_tail_qwen_algo1_public_api_lives_in_package_module() -> None:
    assert (
        collect_qwen_algo1_tail_records.__module__ == "llm_conceptual_modeling.hf_tail.qwen_algo1"
    )
    assert prepare_qwen_algo1_tail_bundle.__module__ == "llm_conceptual_modeling.hf_tail.qwen_algo1"


def test_qwen_algo1_tail_preflight_report_has_explicit_type_contract() -> None:
    hints = get_type_hints(build_qwen_algo1_tail_preflight_report)

    assert hints["return"] is QwenAlgo1TailPreflightReport


def _tail_identity(
    *,
    condition_bits: str = "00101",
    replication: int = 0,
    pair_name: str = "sg1_sg2",
    model: str = QWEN_ALGO1_TAIL_MODEL,
    algorithm: str = "algo1",
    condition_label: str = QWEN_ALGO1_TAIL_CONDITION_LABEL,
) -> dict[str, object]:
    return {
        "algorithm": algorithm,
        "condition_bits": condition_bits,
        "condition_label": condition_label,
        "model": model,
        "pair_name": pair_name,
        "replication": replication,
    }


def _tail_record(
    *,
    status: str = "pending",
    **identity_overrides: object,
) -> dict[str, object]:
    return {"identity": _tail_identity(**identity_overrides), "status": status}


def test_tail_identity_keys_preserve_the_complete_run_identity() -> None:
    identity = _tail_identity(condition_bits="10100", replication=4)
    spec = SimpleNamespace(**identity)

    expected = (
        "algo1",
        QWEN_ALGO1_TAIL_MODEL,
        QWEN_ALGO1_TAIL_CONDITION_LABEL,
        "sg1_sg2",
        "10100",
        4,
    )
    assert tail._tail_identity_key(identity) == expected
    assert tail._planned_spec_identity_key(spec) == expected


def test_tail_watcher_and_manifest_validators_reject_invalid_boundaries() -> None:
    tail._validate_tail_watcher_status({})
    tail._validate_tail_watcher_status({"status": "healthy"})
    with pytest.raises(RuntimeError) as exc_info:
        tail._validate_tail_watcher_status({"status": "degraded"})
    assert str(exc_info.value) == (
        "Dedicated tail watcher is degraded; refusing fresh-host prep."
    )

    tail._validate_tail_manifest_identities([{} for _ in range(10)])
    for invalid in (None, {}, [{} for _ in range(9)], [{} for _ in range(11)]):
        with pytest.raises(RuntimeError) as exc_info:
            tail._validate_tail_manifest_identities(invalid)
        assert str(exc_info.value) == (
            "Dedicated tail manifest does not contain the expected 10 identities."
        )


def test_read_canonical_tail_ledger_requires_a_records_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps({"records": []}), encoding="utf-8")
    read_calls: list[tuple[Path, dict[str, object]]] = []
    real_read_text = Path.read_text

    def read_text(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> str:
        if path == ledger_path:
            read_calls.append((path, dict(kwargs)))
        return real_read_text(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", read_text)
    assert tail._read_canonical_tail_ledger(tmp_path) == {"records": []}

    for payload in ([], {}, {"records": {}}):
        ledger_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(RuntimeError) as exc_info:
            tail._read_canonical_tail_ledger(tmp_path)
        assert str(exc_info.value) == (
            "Canonical ledger.json is missing records; refusing fresh-host prep."
        )
    assert read_calls
    assert all(path == ledger_path for path, _kwargs in read_calls)
    assert all(kwargs == {"encoding": "utf-8"} for _path, kwargs in read_calls)


def test_filter_planned_tail_specs_keeps_only_manifest_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed = SimpleNamespace(**_tail_identity(condition_bits="00101", replication=0))
    excluded = SimpleNamespace(**_tail_identity(condition_bits="10100", replication=0))
    config = SimpleNamespace(
        models=SimpleNamespace(chat_models=["chat"], embedding_model="embed"),
        run=SimpleNamespace(replications=5),
    )
    observed: dict[str, object] = {}

    def fake_plan(**kwargs: object) -> list[object]:
        observed.update(kwargs)
        return [allowed, excluded]

    monkeypatch.setattr(tail, "plan_paper_batch", fake_plan)

    filtered = tail._filter_planned_tail_specs(config, [_tail_identity()])

    assert filtered == [allowed]
    assert observed["models"] == ["chat"]
    assert observed["embedding_model"] == "embed"
    assert observed["replications"] == 5
    assert observed["config"] is config
    assert observed["runtime_profile_provider"] is tail.default_runtime_profile_provider


def test_tail_count_validators_cover_exact_expected_count() -> None:
    tail._validate_planned_tail_specs([object() for _ in range(10)])
    tail._validate_unfinished_tail_count(10)
    for value in (9, 11):
        with pytest.raises(RuntimeError) as planned_exc_info:
            tail._validate_planned_tail_specs([object() for _ in range(value)])
        assert str(planned_exc_info.value) == (
            "Dedicated tail config planned an unexpected number of manifest-matched runs."
        )
        with pytest.raises(RuntimeError) as unfinished_exc_info:
            tail._validate_unfinished_tail_count(value)
        assert str(unfinished_exc_info.value) == (
            "Dedicated tail ledger does not resolve to exactly 10 unfinished runs."
        )


def test_unfinished_tail_count_sums_all_retryable_and_failed_states() -> None:
    ledger = {
        "pending_count": "2",
        "retryable_failed_count": 3,
        "terminal_failed_count": 5,
    }

    assert tail._unfinished_tail_count(ledger) == 10


def test_normalize_tail_record_filters_finished_and_non_target_records() -> None:
    valid = _tail_record(status="retryable_failed", replication=3)
    normalized = tail._normalize_tail_record(valid)
    assert normalized == {"identity": _tail_identity(replication=3), "status": "retryable_failed"}

    default_status = tail._normalize_tail_record({"identity": _tail_identity()})
    assert default_status == {"identity": _tail_identity(), "status": "pending"}
    assert tail._normalize_tail_record(_tail_record(status="finished")) is None
    assert tail._normalize_tail_record(_tail_record(model="other")) is None
    assert tail._normalize_tail_record(_tail_record(algorithm="other")) is None
    assert tail._normalize_tail_record({"identity": None, "status": "pending"}) is None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "pair_name",
            "sg2_sg3",
            "Qwen algo1 tail unexpectedly contains a non-sg1_sg2 pair.",
        ),
        (
            "condition_bits",
            "11111",
            "Qwen algo1 tail unexpectedly contains an unknown condition_bits.",
        ),
        (
            "replication",
            5,
            "Qwen algo1 tail unexpectedly contains an unknown replication.",
        ),
    ],
)
def test_validate_tail_record_identity_rejects_each_unexpected_dimension(
    field: str,
    value: object,
    message: str,
) -> None:
    identity = _tail_identity()
    identity[field] = value

    with pytest.raises(ValueError) as exc_info:
        tail._validate_tail_record_identity(identity)
    assert str(exc_info.value) == message


def test_collect_unique_tail_records_skips_invalid_finished_and_duplicate_records() -> None:
    first = _tail_record(status="pending", replication=0)
    duplicate = _tail_record(status="retryable_failed", replication=0)
    later = _tail_record(status="retryable_failed", replication=1)
    records = tail._collect_unique_tail_records(
        [
            None,
            {"identity": _tail_identity(model="other"), "status": "pending"},
            _tail_record(status="finished", replication=2),
            first,
            duplicate,
            later,
        ]
    )

    assert records == [
        {"identity": _tail_identity(), "status": "pending"},
        {"identity": _tail_identity(replication=1), "status": "retryable_failed"},
    ]


def test_validate_tail_record_count_requires_exactly_ten_records() -> None:
    records = [{"identity": _tail_identity(), "status": "pending"} for _ in range(10)]
    tail._validate_tail_record_count(records)
    for count in (9, 11):
        with pytest.raises(ValueError) as exc_info:
            tail._validate_tail_record_count(records[:count] if count == 9 else records + [{}])
        assert str(exc_info.value) == (
            "Expected exactly 10 unfinished Qwen algo1 tail records, "
            f"found {count}."
        )


def test_manifest_writer_persists_scoped_roots_and_shard_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_root = tmp_path / "canonical"
    tail_root = tmp_path / "tail"
    tail_root.mkdir()
    records = [{"identity": _tail_identity(), "status": "pending"}]
    write_calls: list[tuple[Path, dict[str, object]]] = []
    real_write_text = Path.write_text

    def write_text(
        path: Path,
        data: str,
        *args: object,
        **kwargs: object,
    ) -> int:
        write_calls.append((path, dict(kwargs)))
        return real_write_text(path, data, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "write_text", write_text)

    manifest = tail.write_qwen_algo1_tail_manifest(
        canonical_results_root=canonical_root,
        tail_results_root=tail_root,
        records=records,
    )

    assert manifest["canonical_results_root"] == str(canonical_root.resolve())
    assert manifest["results_root"] == str(tail_root.resolve())
    assert manifest["ledger_root"] == str(tail_root.resolve())
    assert manifest["active_chat_models"] == [QWEN_ALGO1_TAIL_MODEL]
    assert manifest["shard_count"] == 1
    assert manifest["shard_index"] == 0
    assert manifest["identities"] == [_tail_identity()]
    assert manifest["generated_at"].endswith("+00:00")
    expected_text = json.dumps(manifest, indent=2, sort_keys=True)
    assert (tail_root / "shard_manifest.json").read_text(encoding="utf-8") == expected_text
    assert json.loads(expected_text) == manifest
    assert write_calls == [(tail_root / "shard_manifest.json", {"encoding": "utf-8"})]


def test_seed_ledger_writer_persists_all_initial_state_counters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tail_root = tmp_path / "tail"
    tail_root.mkdir()
    records = [
        {
            "status": "retryable_failed",
            "identity": {
                "replication": 0,
                "pair_name": "sg1_sg2",
                "model": QWEN_ALGO1_TAIL_MODEL,
                "condition_label": QWEN_ALGO1_TAIL_CONDITION_LABEL,
                "condition_bits": "00101",
                "algorithm": "algo1",
            },
        }
    ]
    write_calls: list[tuple[Path, dict[str, object]]] = []
    real_write_text = Path.write_text

    def write_text(
        path: Path,
        data: str,
        *args: object,
        **kwargs: object,
    ) -> int:
        write_calls.append((path, dict(kwargs)))
        return real_write_text(path, data, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "write_text", write_text)

    ledger = tail.write_qwen_algo1_tail_seed_ledger(tail_results_root=tail_root, records=records)

    assert ledger["duplicate_extra_artifact_count"] == 0
    assert ledger["duplicate_logical_run_count"] == 0
    assert ledger["expected_total_runs"] == 1
    assert ledger["finished_count"] == 0
    assert ledger["pending_count"] == 1
    assert ledger["retryable_failed_count"] == 1
    assert ledger["terminal_failed_count"] == 0
    assert ledger["records"] == records
    assert ledger["generated_at"].endswith("+00:00")
    expected_text = json.dumps(ledger, indent=2, sort_keys=True)
    assert (tail_root / "ledger.json").read_text(encoding="utf-8") == expected_text
    assert json.loads(expected_text) == ledger
    assert write_calls == [(tail_root / "ledger.json", {"encoding": "utf-8"})]


def test_read_watcher_status_returns_empty_for_missing_and_payload_for_existing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing.json"
    assert tail.read_watcher_status(missing) == {}
    status_path = tmp_path / "status.json"
    status_path.write_text(json.dumps({"status": "healthy", "pid": 7}), encoding="utf-8")
    read_calls: list[tuple[Path, dict[str, object]]] = []
    real_read_text = Path.read_text

    def read_text(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> str:
        if path == status_path:
            read_calls.append((path, dict(kwargs)))
        return real_read_text(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", read_text)
    assert tail.read_watcher_status(status_path) == {"status": "healthy", "pid": 7}
    assert read_calls == [(status_path.resolve(), {"encoding": "utf-8"})]


def test_ledger_snapshot_coerces_values_and_supplies_zero_defaults() -> None:
    assert tail._ledger_snapshot(
        {
            "finished_count": "1",
            "pending_count": 2,
            "retryable_failed_count": "3",
            "terminal_failed_count": 4,
            "expected_total_runs": "10",
        }
    ) == {
        "finished_count": 1,
        "pending_count": 2,
        "retryable_failed_count": 3,
        "terminal_failed_count": 4,
        "expected_total_runs": 10,
    }
    assert tail._ledger_snapshot({}) == {
        "finished_count": 0,
        "pending_count": 0,
        "retryable_failed_count": 0,
        "terminal_failed_count": 0,
        "expected_total_runs": 0,
    }


def test_tail_run_dir_and_json_readers_preserve_paths_and_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _tail_identity(condition_bits="10100", replication=4)
    expected_run_dir = (
        tmp_path
        / "runs"
        / "algo1"
        / "Qwen__Qwen3.5-9B"
        / QWEN_ALGO1_TAIL_CONDITION_LABEL
        / "sg1_sg2"
        / "10100"
        / "rep_04"
    )
    assert tail._tail_run_dir(tmp_path, identity) == expected_run_dir

    manifest = {"identities": [identity], "shard_count": 1}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps({"records": []}), encoding="utf-8")
    read_calls: list[tuple[Path, dict[str, object]]] = []
    real_read_text = Path.read_text

    def read_text(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> str:
        if path in {manifest_path, ledger_path}:
            read_calls.append((path, dict(kwargs)))
        return real_read_text(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", read_text)
    assert tail._read_tail_manifest(manifest_path) == manifest
    assert tail._read_tail_seed_ledger(ledger_path) == {"records": []}
    assert read_calls == [
        (manifest_path, {"encoding": "utf-8"}),
        (ledger_path, {"encoding": "utf-8"}),
    ]
