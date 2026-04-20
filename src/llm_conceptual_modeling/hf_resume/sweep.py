from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Callable, TypedDict

from llm_conceptual_modeling.common.io import coerce_int
from llm_conceptual_modeling.hf_config.run_config import load_hf_run_config
from llm_conceptual_modeling.hf_resume.preflight import build_resume_preflight_report
from llm_conceptual_modeling.hf_resume.profile import resolve_resume_profile


class ResumeRootReport(TypedDict, total=False):
    results_root: str
    config_source: str | None
    classification: str
    can_resume: bool
    resume_mode: object
    total_runs: int
    finished_count: int
    failed_count: int
    pending_count: int
    running_count: int
    status_updated_at: object
    ready_to_rent: bool
    rent_ready: bool
    rent_ready_reason: str
    needs_config_fix: bool
    active: bool
    runtime_mode: str
    resume_profile: str
    resume_phase: str
    excluded_decoding_labels: list[str]
    error: str


class ResumeSweepReport(TypedDict):
    repo_root: str
    results_root: str
    root_count: int
    ready_count: int
    needs_config_fix_count: int
    active_count: int
    invalid_config_count: int
    missing_config_count: int
    finished_count: int
    recommended_results_root: str | None
    recommended_reason: str | None
    roots: list[ResumeRootReport]


ResumeConfigLoader = Callable[[Path], object]
ResumePreflightBuilder = Callable[..., Mapping[str, object]]


def build_resume_sweep_report(
    *,
    repo_root: Path,
    results_root: Path,
    load_config_fn: ResumeConfigLoader = load_hf_run_config,
    preflight_fn: ResumePreflightBuilder = build_resume_preflight_report,
    root_name_contains: str | None = None,
) -> ResumeSweepReport:
    repo_root_path = Path(repo_root)
    results_root_path = Path(results_root)
    if not repo_root_path.exists():
        raise FileNotFoundError(f"Local repo root does not exist: {repo_root_path}")
    if not results_root_path.exists():
        raise FileNotFoundError(f"Local results root does not exist: {results_root_path}")

    root_reports: list[ResumeRootReport] = []
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
            mismatch_error = _build_output_root_mismatch_error(
                results_root=candidate_root,
                config=config,
            )
            if mismatch_error is not None:
                raise ValueError(mismatch_error)
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
        can_resume = _report_bool(report, "can_resume")
        root_reports.append(
            {
                "results_root": str(candidate_root),
                "config_source": str(config_source),
                "classification": classification,
                "can_resume": can_resume,
                "resume_mode": report.get("resume_mode"),
                "total_runs": coerce_int(report.get("total_runs", 0)),
                "finished_count": coerce_int(report.get("finished_count", 0)),
                "failed_count": coerce_int(report.get("failed_count", 0)),
                "pending_count": coerce_int(report.get("pending_count", 0)),
                "running_count": coerce_int(report.get("running_count", 0)),
                "status_updated_at": report.get("status_updated_at"),
                "ready_to_rent": classification == "resume-ready",
                "rent_ready": classification == "resume-ready",
                "rent_ready_reason": classification,
                "needs_config_fix": classification == "needs-config-fix",
                "active": classification == "active",
                "runtime_mode": resume_profile.runtime_mode,
                "resume_profile": resume_profile.profile_name,
                "resume_phase": resume_profile.phase,
                "excluded_decoding_labels": list(resume_profile.excluded_decoding_labels),
            }
        )

    ordered_roots = sorted(root_reports, key=_root_sort_key)
    recommended_root = _recommend_root_report(ordered_roots)
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
        "recommended_results_root": (
            str(recommended_root["results_root"]) if recommended_root is not None else None
        ),
        "recommended_reason": (
            "largest rent-ready backlog" if recommended_root is not None else None
        ),
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


def _classify_root_report(report: Mapping[str, object]) -> str:
    running_count = coerce_int(report.get("running_count", 0))
    if running_count > 0:
        return "active"
    if _report_bool(report, "can_resume"):
        return "resume-ready"
    if coerce_int(report.get("failed_count", 0)) > 0:
        return "needs-config-fix"
    return "finished"


def _build_output_root_mismatch_error(*, results_root: Path, config: object) -> str | None:
    configured_output_root = _configured_output_root_name(config)
    if configured_output_root is None:
        return None
    if configured_output_root == results_root.name:
        return None
    return (
        "Configured output root "
        f"{configured_output_root!r} does not match results root {results_root.name!r}."
    )


def _configured_output_root_name(config: object) -> str | None:
    run_config = getattr(config, "run", None)
    output_root = getattr(run_config, "output_root", None)
    if not output_root:
        return None
    return Path(str(output_root)).name


def _root_sort_key(report: ResumeRootReport) -> tuple[int, str]:
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


def _recommend_root_report(root_reports: list[ResumeRootReport]) -> ResumeRootReport | None:
    rent_ready_roots = [report for report in root_reports if _report_bool(report, "rent_ready")]
    if not rent_ready_roots:
        return None
    return max(
        rent_ready_roots,
        key=lambda report: (
            coerce_int(report.get("pending_count", 0)),
            coerce_int(report.get("failed_count", 0)),
            -coerce_int(report.get("finished_count", 0)),
            str(report.get("results_root", "")),
        ),
    )


def _report_bool(report: Mapping[str, object], key: str) -> bool:
    return report.get(key) is True
