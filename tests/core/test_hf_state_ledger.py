from pathlib import Path

from llm_conceptual_modeling.hf_state.ledger import refresh_ledger


def test_hf_state_ledger_package_imports_and_refreshes(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    ledger_root = results_root / "hf-paper-batch-canonical"
    ledger_root.mkdir(parents=True, exist_ok=True)
    (ledger_root / "ledger.json").write_text(
        """
        {
          "records": [
            {
              "identity": {
                "algorithm": "algo1",
                "condition_bits": "00000",
                "condition_label": "greedy",
                "model": "Qwen/Qwen3.5-9B",
                "pair_name": "sg1_sg2",
                "replication": 0
              },
              "status": "pending"
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    refreshed = refresh_ledger(results_root=results_root, ledger_root=ledger_root)

    assert refreshed["expected_total_runs"] == 1
    assert refreshed["finished_count"] == 0
    assert refreshed["pending_count"] == 1
