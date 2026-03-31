import json
from pathlib import Path

import pandas as pd
import pytest

from llm_conceptual_modeling.hf_experiments import plan_paper_batch, run_paper_batch
from llm_conceptual_modeling.hf_run_config import load_hf_run_config


def test_plan_paper_batch_covers_full_factorial_surface() -> None:
    specs = plan_paper_batch(
        models=["mistralai/Ministral-3-8B-Instruct-2512"],
        embedding_model="Qwen/Qwen3-Embedding-8B",
        replications=1,
    )

    assert len(specs) == 1680


def test_run_paper_batch_writes_resumable_state_and_manifest(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"

    def runtime_factory(spec):
        if spec.algorithm == "algo3":
            row = {
                **spec.raw_context,
                "Results": "[]",
                "Source Graph": "[]",
                "Target Graph": "[]",
                "Mother Graph": "[]",
                "Recall": 0.0,
            }
        else:
            row = {
                **spec.raw_context,
                "Result": "[]",
                "graph": "[]",
                "subgraph1": "[]",
                "subgraph2": "[]",
            }
        return {
            "raw_row": row,
            "runtime": {"thinking_mode_supported": False},
            "raw_response": "{}",
        }

    run_paper_batch(
        output_root=output_root,
        models=["mistralai/Ministral-3-8B-Instruct-2512"],
        embedding_model="Qwen/Qwen3-Embedding-8B",
        replications=1,
        runtime_factory=runtime_factory,
    )

    manifest_paths = list(output_root.rglob("manifest.json"))
    state_paths = list(output_root.rglob("state.json"))
    summary_paths = list(output_root.rglob("summary.json"))

    assert manifest_paths
    assert state_paths
    assert summary_paths

    manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
    assert manifest["provider"] == "hf-transformers"
    assert manifest["embedding_model"] == "Qwen/Qwen3-Embedding-8B"
    assert manifest["decoding"]["algorithm"] in {"greedy", "beam", "contrastive"}
    assert manifest["runtime"]["device"] == "cuda"
    assert manifest["runtime"]["quantization"] == "none"


def test_run_paper_batch_resumes_without_recomputing_finished_runs(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"
    call_count = {"count": 0}

    def runtime_factory(spec):
        call_count["count"] += 1
        if spec.algorithm == "algo3":
            row = {
                **spec.raw_context,
                "Results": "[]",
                "Source Graph": "[]",
                "Target Graph": "[]",
                "Mother Graph": "[]",
                "Recall": 0.0,
            }
        else:
            row = {
                **spec.raw_context,
                "Result": "[]",
                "graph": "[]",
                "subgraph1": "[]",
                "subgraph2": "[]",
            }
        return {
            "raw_row": row,
            "runtime": {"thinking_mode_supported": False},
            "raw_response": "{}",
        }

    run_paper_batch(
        output_root=output_root,
        models=["mistralai/Ministral-3-8B-Instruct-2512"],
        embedding_model="Qwen/Qwen3-Embedding-8B",
        replications=1,
        runtime_factory=runtime_factory,
    )
    first_count = call_count["count"]

    run_paper_batch(
        output_root=output_root,
        models=["mistralai/Ministral-3-8B-Instruct-2512"],
        embedding_model="Qwen/Qwen3-Embedding-8B",
        replications=1,
        runtime_factory=runtime_factory,
        resume=True,
    )

    assert call_count["count"] == first_count


def test_run_paper_batch_writes_batch_summary_csv(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"

    def runtime_factory(spec):
        if spec.algorithm == "algo3":
            row = {
                **spec.raw_context,
                "Results": "[]",
                "Source Graph": "[]",
                "Target Graph": "[]",
                "Mother Graph": "[]",
                "Recall": 0.0,
            }
        else:
            row = {
                **spec.raw_context,
                "Result": "[]",
                "graph": "[]",
                "subgraph1": "[]",
                "subgraph2": "[]",
            }
        return {
            "raw_row": row,
            "runtime": {"thinking_mode_supported": spec.model.startswith("mistralai/")},
            "raw_response": "{}",
        }

    run_paper_batch(
        output_root=output_root,
        models=["mistralai/Ministral-3-8B-Instruct-2512"],
        embedding_model="Qwen/Qwen3-Embedding-8B",
        replications=1,
        runtime_factory=runtime_factory,
    )

    summary = pd.read_csv(output_root / "batch_summary.csv")

    assert {"algorithm", "model", "decoding_algorithm", "replication", "status"}.issubset(
        summary.columns
    )


def test_run_paper_batch_writes_error_artifact_and_marks_failed_state(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"

    with pytest.raises(RuntimeError, match="boom"):
        run_paper_batch(
            output_root=output_root,
            models=["mistralai/Ministral-3-8B-Instruct-2512"],
            embedding_model="Qwen/Qwen3-Embedding-8B",
            replications=1,
            runtime_factory=lambda _spec: (_ for _ in ()).throw(RuntimeError("boom")),
        )

    error_paths = list(output_root.rglob("error.json"))
    state_paths = list(output_root.rglob("state.json"))

    assert error_paths
    assert any(
        json.loads(path.read_text(encoding="utf-8")).get("status") == "failed"
        for path in state_paths
    )


def test_run_paper_batch_writes_aggregated_outputs_and_ci_reports(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"

    def runtime_factory(spec):
        if spec.algorithm == "algo3":
            row = {
                **spec.raw_context,
                "Results": "[]",
                "Source Graph": "[]",
                "Target Graph": "[]",
                "Mother Graph": "[]",
                "Recall": 0.0,
            }
        else:
            row = {
                **spec.raw_context,
                "Result": "[]",
                "graph": "[]",
                "subgraph1": "[]",
                "subgraph2": "[]",
            }
        return {
            "raw_row": row,
            "runtime": {"thinking_mode_supported": False},
            "raw_response": "{}",
        }

    run_paper_batch(
        output_root=output_root,
        models=["mistralai/Ministral-3-8B-Instruct-2512"],
        embedding_model="Qwen/Qwen3-Embedding-8B",
        replications=2,
        runtime_factory=runtime_factory,
    )

    assert list((output_root / "aggregated").rglob("raw.csv"))
    assert list((output_root / "aggregated").rglob("evaluated.csv"))
    assert list((output_root / "aggregated").rglob("factorial.csv"))
    assert list((output_root / "aggregated").rglob("replication_budget_strict.csv"))
    assert list((output_root / "aggregated").rglob("replication_budget_relaxed.csv"))


def test_run_paper_batch_uses_yaml_config_as_execution_source_of_truth(tmp_path: Path) -> None:
    config_path = tmp_path / "paper_batch.yaml"
    configured_output_root = tmp_path / "configured-runs"
    config_path.write_text(
        f"""
run:
  provider: hf-transformers
  output_root: {configured_output_root}
  replications: 1
runtime:
  seed: 11
  temperature: 1.0
  quantization: none
  device_policy: cuda-only
  context_policy:
    prompt_truncation: forbid
    safety_margin_tokens: 32
  max_new_tokens_by_schema:
    edge_list: 256
    vote_list: 64
    label_list: 128
    children_by_label: 384
models:
  chat_models:
    - mistralai/Ministral-3-8B-Instruct-2512
  embedding_model: Qwen/Qwen3-Embedding-8B
decoding:
  greedy:
    enabled: true
inputs:
  graph_source: default
shared_fragments:
  assistant_role: "You are a helpful assistant."
algorithms:
  algo1:
    pair_names: [sg1_sg2]
    base_fragments: [assistant_role]
    factors:
      explanation:
        column: Explanation
        levels: [-1, 1]
        runtime_field: include_explanation
        low_runtime_value: false
        high_runtime_value: true
        low_fragments: []
        high_fragments: [explanation_text]
      example:
        column: Example
        levels: [-1, 1]
        runtime_field: include_example
        low_runtime_value: false
        high_runtime_value: true
        low_fragments: []
        high_fragments: []
      counterexample:
        column: Counterexample
        levels: [-1, 1]
        runtime_field: include_counterexample
        low_runtime_value: false
        high_runtime_value: true
        low_fragments: []
        high_fragments: []
      array_repr:
        column: Array/List(1/-1)
        levels: [-1, 1]
        runtime_field: use_array_representation
        low_runtime_value: false
        high_runtime_value: true
        low_fragments: []
        high_fragments: []
      adjacency_repr:
        column: Tag/Adjacency(1/-1)
        levels: [-1, 1]
        runtime_field: use_adjacency_notation
        low_runtime_value: false
        high_runtime_value: true
        low_fragments: []
        high_fragments: []
    fragment_definitions:
      explanation_text: "Explain the notation."
    prompt_templates:
      direct_edge: >-
        Knowledge map 1: {{formatted_subgraph1}} Knowledge map 2: {{formatted_subgraph2}}
      cove_verification: "Candidate pairs: {{candidate_edges}}"
""",
        encoding="utf-8",
    )
    config = load_hf_run_config(config_path)

    run_paper_batch(
        output_root=tmp_path / "ignored",
        models=["ignored/model"],
        embedding_model="ignored/embedding",
        replications=99,
        config=config,
        dry_run=True,
    )

    summary = pd.read_csv(configured_output_root / "batch_summary.csv")
    manifest_path = next(configured_output_root.rglob("manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert summary["algorithm"].unique().tolist() == ["algo1"]
    assert manifest["temperature"] == 1.0
    assert manifest["base_seed"] == 11
    assert isinstance(manifest["seed"], int)
