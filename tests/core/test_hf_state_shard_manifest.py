from pathlib import Path

from llm_conceptual_modeling.hf_state.shard_manifest import (
    manifest_identity_keys,
    write_unfinished_shard_manifest,
)


def test_hf_state_shard_manifest_package_imports_and_writes_manifest(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    ledger_root = results_root / "hf-paper-batch-canonical"
    batch_root = results_root / "hf-paper-batch-canonical"
    (ledger_root).mkdir(parents=True, exist_ok=True)
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
    (batch_root / "runtime_config.yaml").write_text(
        "\n".join(
            [
                "models:",
                "  chat_models:",
                "    - Qwen/Qwen3.5-9B",
            ]
        ),
        encoding="utf-8",
    )

    manifest = write_unfinished_shard_manifest(results_root=results_root, ledger_root=ledger_root)

    assert manifest["shard_count"] == 1
    assert manifest_identity_keys(manifest) == {
        ("algo1", "Qwen/Qwen3.5-9B", "greedy", "sg1_sg2", "00000", 0)
    }


def test_write_unfinished_shard_manifest_preserves_graph_source_identity(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    ledger_root = results_root / "hf-paper-batch-canonical"
    batch_root = results_root / "hf-paper-batch-canonical"
    ledger_root.mkdir(parents=True, exist_ok=True)
    (ledger_root / "ledger.json").write_text(
        """
        {
          "records": [
            {
              "identity": {
                "algorithm": "algo3",
                "condition_bits": "000",
                "condition_label": "beam_num_beams_6",
                "graph_source": "babs_johnson",
                "model": "Qwen/Qwen3.5-9B",
                "pair_name": "subgraph_1_to_subgraph_3",
                "replication": 0
              },
              "status": "pending"
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )
    (batch_root / "runtime_config.yaml").write_text(
        "\n".join(
            [
                "models:",
                "  chat_models:",
                "    - Qwen/Qwen3.5-9B",
            ]
        ),
        encoding="utf-8",
    )

    manifest = write_unfinished_shard_manifest(results_root=results_root, ledger_root=ledger_root)

    assert manifest["identities"] == [
        {
            "algorithm": "algo3",
            "condition_bits": "000",
            "condition_label": "beam_num_beams_6",
            "graph_source": "babs_johnson",
            "model": "Qwen/Qwen3.5-9B",
            "pair_name": "subgraph_1_to_subgraph_3",
            "replication": 0,
        }
    ]
    assert manifest_identity_keys(manifest) == {
        (
            "algo3",
            "Qwen/Qwen3.5-9B",
            "beam_num_beams_6",
            "babs_johnson",
            "subgraph_1_to_subgraph_3",
            "000",
            0,
        )
    }
