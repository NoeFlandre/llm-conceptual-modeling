from __future__ import annotations

import json
from pathlib import Path

from llm_conceptual_modeling.hf_state.shard_manifest import write_unfinished_shard_manifest


def test_write_unfinished_shard_manifest_keeps_only_active_unfinished_identities(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "runtime_config.yaml").write_text(
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
    ledger_root = results_root
    (ledger_root / "ledger.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "identity": {
                            "algorithm": "algo1",
                            "condition_bits": "00000",
                            "condition_label": "greedy",
                            "model": "Qwen/Qwen3.5-9B",
                            "pair_name": "sg1_sg2",
                            "replication": 0,
                        },
                        "status": "finished",
                    },
                    {
                        "identity": {
                            "algorithm": "algo2",
                            "condition_bits": "000001",
                            "condition_label": "beam_num_beams_2",
                            "model": "mistralai/Ministral-3-8B-Instruct-2512",
                            "pair_name": "sg3_sg1",
                            "replication": 1,
                        },
                        "status": "retryable_failed",
                    },
                    {
                        "identity": {
                            "algorithm": "algo3",
                            "condition_bits": "0010",
                            "condition_label": "greedy",
                            "model": "allenai/Olmo-3-7B-Instruct",
                            "pair_name": "subgraph_2_to_subgraph_3",
                            "replication": 2,
                        },
                        "status": "terminal_failed",
                    },
                    {
                        "identity": {
                            "algorithm": "algo1",
                            "condition_bits": "11111",
                            "condition_label": "contrastive_penalty_alpha_0.8",
                            "model": "Qwen/Qwen3.5-9B",
                            "pair_name": "sg2_sg3",
                            "replication": 4,
                        },
                        "status": "pending",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    manifest = write_unfinished_shard_manifest(
        results_root=results_root,
        ledger_root=ledger_root,
    )

    identities = manifest["identities"]
    assert len(identities) == 2
    assert {identity["model"] for identity in identities} == {
        "mistralai/Ministral-3-8B-Instruct-2512",
        "Qwen/Qwen3.5-9B",
    }
    assert {identity["condition_bits"] for identity in identities} == {"000001", "11111"}
