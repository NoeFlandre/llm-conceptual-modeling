import json
from pathlib import Path

import pandas as pd
import pytest

from llm_conceptual_modeling.hf_experiments import plan_paper_batch, run_paper_batch


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
