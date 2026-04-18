from __future__ import annotations

import json
from pathlib import Path

from llm_conceptual_modeling.hf_state.ledger import refresh_ledger


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_finished_run(run_dir: Path) -> None:
    _write_json(run_dir / "state.json", {"status": "finished"})
    _write_json(
        run_dir / "summary.json",
        {
            "status": "finished",
            "algorithm": "algo1",
            "model": "Qwen/Qwen3.5-9B",
            "condition_label": "greedy",
            "condition_bits": "00000",
            "pair_name": "sg1_sg2",
            "replication": 0,
            "accuracy": 0.75,
            "precision": 0.8,
            "recall": 0.75,
            "f1": 0.77,
        },
    )
    _write_json(
        run_dir / "raw_row.json",
        {
            "status": "finished",
            "algorithm": "algo1",
            "model": "Qwen/Qwen3.5-9B",
            "condition_label": "greedy",
            "condition_bits": "00000",
            "pair_name": "sg1_sg2",
            "replication": 0,
        },
    )
    _write_json(run_dir / "manifest.json", {"status": "finished"})
    _write_json(run_dir / "runtime.json", {"status": "finished"})


def _write_failed_run(run_dir: Path, *, message: str, error_type: str = "RuntimeError") -> None:
    _write_json(run_dir / "state.json", {"status": "failed"})
    _write_json(
        run_dir / "error.json",
        {
            "type": error_type,
            "message": message,
            "status": "failed",
        },
    )


def _write_runtime_config(path: Path) -> None:
    path.write_text(
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


def test_refresh_ledger_rebuilds_counts_from_multiple_roots(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    ledger_root = results_root / "hf-paper-batch-canonical"
    ledger_root.mkdir(parents=True, exist_ok=True)

    identity_finished = {
        "algorithm": "algo1",
        "condition_bits": "00000",
        "condition_label": "greedy",
        "model": "Qwen/Qwen3.5-9B",
        "pair_name": "sg1_sg2",
        "replication": 0,
    }
    identity_pending = {
        "algorithm": "algo1",
        "condition_bits": "00001",
        "condition_label": "greedy",
        "model": "Qwen/Qwen3.5-9B",
        "pair_name": "sg1_sg2",
        "replication": 1,
    }
    identity_retryable = {
        "algorithm": "algo1",
        "condition_bits": "00010",
        "condition_label": "greedy",
        "model": "Qwen/Qwen3.5-9B",
        "pair_name": "sg1_sg2",
        "replication": 2,
    }

    _write_json(
        ledger_root / "ledger.json",
        {
            "duplicate_extra_artifact_count": 0,
            "duplicate_logical_run_count": 0,
            "expected_total_runs": 2,
            "finished_count": 0,
            "generated_at": "2026-04-10T00:00:00+00:00",
            "pending_count": 2,
            "records": [
                {
                    "identity": identity_finished,
                    "status": "pending",
                    "candidates": [],
                    "canonical_unresolved": None,
                    "winner": None,
                },
                {
                    "identity": identity_retryable,
                    "status": "pending",
                    "candidates": [],
                    "canonical_unresolved": None,
                    "winner": None,
                },
                {
                    "identity": identity_pending,
                    "status": "pending",
                    "candidates": [],
                    "canonical_unresolved": None,
                    "winner": None,
                },
            ],
            "retryable_failed_count": 0,
            "terminal_failed_count": 0,
        },
    )

    finished_root = results_root / "hf-paper-batch-canonical-current"
    failed_root = results_root / "hf-paper-batch-canonical-redistributed-shard-00-of-04"

    _write_finished_run(
        finished_root
        / "runs"
        / "algo1"
        / "Qwen__Qwen3.5-9B"
        / "greedy"
        / "sg1_sg2"
        / "00000"
        / "rep_00"
    )
    _write_failed_run(
        failed_root
        / "runs"
        / "algo1"
        / "Qwen__Qwen3.5-9B"
        / "greedy"
        / "sg1_sg2"
        / "00000"
        / "rep_00",
        message="Persistent HF worker exited before writing a result artifact.",
        error_type="RuntimeError",
    )

    _write_failed_run(
        failed_root
        / "runs"
        / "algo1"
        / "Qwen__Qwen3.5-9B"
        / "greedy"
        / "sg1_sg2"
        / "00010"
        / "rep_02",
        message="Monitored command exceeded stage timeout of 20.0 seconds.",
        error_type="MonitoredCommandTimeout",
    )

    refreshed = refresh_ledger(results_root=results_root, ledger_root=ledger_root)

    assert refreshed["finished_count"] == 1
    assert refreshed["pending_count"] == 1
    assert refreshed["retryable_failed_count"] == 1
    assert refreshed["terminal_failed_count"] == 0
    assert refreshed["records"][0]["status"] == "finished"
    assert refreshed["records"][0]["winner"]["source_run_dir"].endswith(
        "hf-paper-batch-canonical-current/runs/algo1/Qwen__Qwen3.5-9B/greedy/sg1_sg2/00000/rep_00"
    )
    assert refreshed["records"][1]["status"] == "retryable_failed"
    assert refreshed["records"][1]["canonical_unresolved"]["failure_kind"] == "timeout"
    assert refreshed["records"][2]["status"] == "pending"

    ledger_payload = json.loads((ledger_root / "ledger.json").read_text(encoding="utf-8"))
    assert ledger_payload["finished_count"] == 1
    assert ledger_payload["pending_count"] == 1
    assert ledger_payload["records"][0]["status"] == "finished"


def test_refresh_ledger_distinguishes_graph_sources_in_identity(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    ledger_root = results_root / "hf-paper-batch-canonical"
    ledger_root.mkdir(parents=True, exist_ok=True)

    identity_babs = {
        "algorithm": "algo3",
        "condition_bits": "000",
        "condition_label": "beam_num_beams_6",
        "graph_source": "babs_johnson",
        "model": "Qwen/Qwen3.5-9B",
        "pair_name": "subgraph_1_to_subgraph_3",
        "replication": 0,
    }
    identity_clarice = {
        **identity_babs,
        "graph_source": "clarice_starling",
    }

    _write_json(
        ledger_root / "ledger.json",
        {
            "records": [
                {"identity": identity_babs, "status": "pending"},
                {"identity": identity_clarice, "status": "pending"},
            ]
        },
    )

    batch_root = results_root / "hf-paper-batch-canonical-current"
    _write_finished_run(
        batch_root
        / "runs"
        / "algo3"
        / "Qwen__Qwen3.5-9B"
        / "beam_num_beams_6"
        / "babs_johnson"
        / "subgraph_1_to_subgraph_3"
        / "000"
        / "rep_00"
    )

    refreshed = refresh_ledger(results_root=results_root, ledger_root=ledger_root)

    assert refreshed["finished_count"] == 1
    assert refreshed["pending_count"] == 1
    assert [record["identity"]["graph_source"] for record in refreshed["records"]] == [
        "babs_johnson",
        "clarice_starling",
    ]
    assert [record["status"] for record in refreshed["records"]] == ["finished", "pending"]


def test_hf_ledger_public_api_lives_in_state_module() -> None:
    assert refresh_ledger.__module__ == "llm_conceptual_modeling.hf_state.ledger"
