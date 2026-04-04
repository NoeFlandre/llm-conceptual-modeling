from __future__ import annotations

from pathlib import Path
from typing import Callable

from llm_conceptual_modeling.hf_resume_preflight import build_resume_preflight_report
from llm_conceptual_modeling.hf_resume_profile import resolve_resume_profile
from llm_conceptual_modeling.hf_run_config import (
    HFRunConfig,
    load_hf_run_config,
)

ResumeConfigLoader = Callable[[Path], HFRunConfig]
ResumePreflightBuilder = Callable[..., dict[str, object]]


def build_resume_sweep_report(
    *,
    repo_root: Path,
    results_root: Path,
    load_config_fn: ResumeConfigLoader = load_hf_run_config,
    preflight_fn: ResumePreflightBuilder = build_resume_preflight_report,
    root_name_contains: str | None = None,
) -> dict[str, object]:
    repo_root_path = Path(repo_root)
    results_root_path = Path(results_root)
    if not repo_root_path.exists():
        raise FileNotFoundError(f"Local repo root does not exist: {repo_root_path}")
    if not results_root_path.exists():
        raise FileNotFoundError(f"Local results root does not exist: {results_root_path}")

    root_reports: list[dict[str, object]] = []
    for candidate_root in _discover_resume_roots(results_root_path):
        if root_name_contains is not None and root_name_contains not in candidate_root.name:
            continue
        config_source = resolve_resume_config_source(candidate_root)
        if config_source is None:
            root_reports.append(
                {
                    "results_root": str(candidate_root),
                    "classification": "missing-config",
                    "config_source": None,
                    "can_resume": False,
                }
            )
            continue

        try:
            config = load_config_fn(config_source)
            report = preflight_fn(
                config=config,
                repo_root=repo_root_path,
                results_root=candidate_root,
                allow_empty=True,
            )
        except Exception as error:
            root_reports.append(
                {
                    "results_root": str(candidate_root),
                    "config_source": str(config_source),
                    "classification": "invalid-config",
                    "can_resume": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        classification = _classify_root_report(report)
        resume_profile = resolve_resume_profile(candidate_root.name)
        root_reports.append(
            {
                "results_root": str(candidate_root),
                "config_source": str(config_source),
                "classification": classification,
                "can_resume": bool(report.get("can_resume", False)),
                "resume_mode": report.get("resume_mode"),
                "total_runs": report.get("total_runs", 0),
                "finished_count": report.get("finished_count", 0),
                "failed_count": report.get("failed_count", 0),
                "pending_count": report.get("pending_count", 0),
                "ready_to_rent": classification == "resume-ready",
                "rent_ready": classification == "resume-ready",
                "rent_ready_reason": classification,
                "needs_config_fix": classification == "needs-config-fix",
                "active": classification == "active",
                "runtime_mode": resume_profile.runtime_mode,
                "resume_profile": resume_profile.profile_name,
                "excluded_decoding_labels": list(resume_profile.excluded_decoding_labels),
            }
        )

    ordered_roots = sorted(root_reports, key=_root_sort_key)
    return {
        "repo_root": str(repo_root_path),
        "results_root": str(results_root_path),
        "root_count": len(ordered_roots),
        "ready_count": sum(1 for item in ordered_roots if item["classification"] == "resume-ready"),
        "needs_config_fix_count": sum(
            1 for item in ordered_roots if item["classification"] == "needs-config-fix"
        ),
        "active_count": sum(1 for item in ordered_roots if item["classification"] == "active"),
        "invalid_config_count": sum(
            1 for item in ordered_roots if item["classification"] == "invalid-config"
        ),
        "missing_config_count": sum(
            1 for item in ordered_roots if item["classification"] == "missing-config"
        ),
        "finished_count": sum(1 for item in ordered_roots if item["classification"] == "finished"),
        "roots": ordered_roots,
    }


def resolve_resume_config_source(results_root: Path) -> Path | None:
    for candidate in (
        results_root / "preview_resume" / "resolved_run_config.yaml",
        results_root / "runtime_config.yaml",
        results_root / "preview" / "resolved_run_config.yaml",
    ):
        if candidate.exists():
            return candidate
    return None


def _discover_resume_roots(results_root: Path) -> list[Path]:
    return sorted(
        candidate
        for candidate in results_root.glob("hf-paper-batch-*")
        if candidate.is_dir()
        and (
            (candidate / "batch_status.json").exists()
            or resolve_resume_config_source(candidate) is not None
        )
    )


def _classify_root_report(report: dict[str, object]) -> str:
    running_count = int(report.get("running_count", 0))
    if running_count > 0:
        return "active"
    if bool(report.get("can_resume", False)):
        return "resume-ready"
    if int(report.get("failed_count", 0)) > 0:
        return "needs-config-fix"
    return "finished"


def _root_sort_key(report: dict[str, object]) -> tuple[int, str]:
    classification_order = {
        "resume-ready": 0,
        "needs-config-fix": 1,
        "active": 2,
        "finished": 3,
        "missing-config": 4,
        "invalid-config": 5,
    }
    return (
        classification_order.get(str(report.get("classification", "missing-config")), 5),
        str(report.get("results_root", "")),
    )
