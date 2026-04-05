from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_conceptual_modeling.hf_resume_preflight import build_resume_preflight_report


def _config(output_root: Path) -> object:
    return SimpleNamespace(
        run=SimpleNamespace(output_root=str(output_root), replications=5),
        models=SimpleNamespace(
            chat_models=["allenai/Olmo-3-7B-Instruct"],
            embedding_model="Qwen/Qwen3-Embedding-0.6B",
        ),
    )


def test_build_resume_preflight_report_rejects_missing_explicit_results_root(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "data" / "inputs").mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        build_resume_preflight_report(
            config=_config(tmp_path / "unused"),
            repo_root=repo_root,
            results_root=tmp_path / "missing-results",
        )


def test_build_resume_preflight_report_counts_pending_work_from_existing_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "data" / "inputs").mkdir(parents=True)
    results_root = tmp_path / "results"
    results_root.mkdir()

    monkeypatch.setattr(
        "llm_conceptual_modeling.hf_resume_preflight.plan_paper_batch",
        lambda **_kwargs: [object(), object(), object()],
    )
    monkeypatch.setattr(
        "llm_conceptual_modeling.hf_resume_preflight.build_seeded_resume_snapshot",
        lambda **_kwargs: (
            {
                "finished_count": 1,
                "failed_count": 1,
                "pending_count": 1,
            },
            [],
            set(),
            set(),
        ),
    )

    report = build_resume_preflight_report(
        config=_config(results_root),
        repo_root=repo_root,
        results_root=results_root,
    )

    assert report["results_root_exists"] is True
    assert report["finished_count"] == 1
    assert report["failed_count"] == 1
    assert report["pending_count"] == 1
    assert report["can_resume"] is True
    assert report["resume_mode"] == "resume"


def test_build_resume_preflight_report_marks_running_root_as_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "data" / "inputs").mkdir(parents=True)
    results_root = tmp_path / "results"
    results_root.mkdir()
    (results_root / "batch_status.json").write_text(
        json.dumps(
            {
                "running_count": 1,
                "updated_at": "2026-04-04T10:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "llm_conceptual_modeling.hf_resume_preflight.plan_paper_batch",
        lambda **_kwargs: [object(), object(), object()],
    )
    monkeypatch.setattr(
        "llm_conceptual_modeling.hf_resume_preflight.build_seeded_resume_snapshot",
        lambda **_kwargs: (
            {
                "finished_count": 1,
                "failed_count": 0,
                "pending_count": 2,
            },
            [],
            set(),
            set(),
        ),
    )

    report = build_resume_preflight_report(
        config=_config(results_root),
        repo_root=repo_root,
        results_root=results_root,
    )

    assert report["running_count"] == 1
    assert report["status_updated_at"] == "2026-04-04T10:00:00+00:00"
    assert report["can_resume"] is False
    assert report["resume_mode"] == "active"


def test_build_resume_preflight_report_ignores_malformed_running_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "data" / "inputs").mkdir(parents=True)
    results_root = tmp_path / "results"
    results_root.mkdir()
    (results_root / "batch_status.json").write_text(
        json.dumps(
            {
                "running_count": "false",
                "updated_at": "2026-04-04T10:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "llm_conceptual_modeling.hf_resume_preflight.plan_paper_batch",
        lambda **_kwargs: [object(), object(), object()],
    )
    monkeypatch.setattr(
        "llm_conceptual_modeling.hf_resume_preflight.build_seeded_resume_snapshot",
        lambda **_kwargs: (
            {
                "finished_count": 1,
                "failed_count": 1,
                "pending_count": 1,
            },
            [],
            set(),
            set(),
        ),
    )

    report = build_resume_preflight_report(
        config=_config(results_root),
        repo_root=repo_root,
        results_root=results_root,
    )

    assert report["running_count"] == 0
    assert report["can_resume"] is True
    assert report["resume_mode"] == "resume"
