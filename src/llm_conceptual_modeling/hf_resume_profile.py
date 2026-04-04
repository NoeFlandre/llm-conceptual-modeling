from dataclasses import dataclass

SAFE_PHASE = "safe"
RISKY_PHASE = "risky"


@dataclass(frozen=True)
class ResumeProfile:
    profile_name: str
    phase: str
    runtime_mode: str
    excluded_decoding_labels: tuple[str, ...]
    retry_timeout_failures_on_resume: bool
    retry_oom_failures_on_resume: bool
    retry_infrastructure_failures_on_resume: bool
    retry_structural_failures_on_resume: bool
    generation_timeout_seconds: int
    startup_timeout_seconds: int
    worker_process_mode: str
    max_requests_per_worker_process: int


def resolve_resume_profile(
    results_root_name: str,
    *,
    phase: str = SAFE_PHASE,
    full_coverage: bool = False,
) -> ResumeProfile:
    if full_coverage:
        return _build_profile(
            profile_name="full-coverage",
            phase=RISKY_PHASE,
            excluded_decoding_labels=(),
        )

    normalized_name = results_root_name.lower()
    normalized_phase = _normalize_phase(phase)
    if "olmo" in normalized_name:
        return _build_olmo_profile(normalized_phase)
    if "qwen" in normalized_name:
        return _build_qwen_profile(normalized_phase)
    if "mistral" in normalized_name:
        return _build_mistral_profile(normalized_phase)
    return _build_profile(
        profile_name=f"default-{normalized_phase}",
        phase=normalized_phase,
        excluded_decoding_labels=(),
    )


def _normalize_phase(phase: str) -> str:
    normalized_phase = phase.strip().lower()
    if normalized_phase not in {SAFE_PHASE, RISKY_PHASE}:
        raise ValueError(f"Unsupported resume profile phase: {phase!r}")
    return normalized_phase


def _build_olmo_profile(phase: str) -> ResumeProfile:
    if phase == SAFE_PHASE:
        return _build_profile(
            profile_name="olmo-safe",
            phase=phase,
            excluded_decoding_labels=("contrastive_penalty_alpha_0.8",),
        )
    return _build_profile(
        profile_name="olmo-risky",
        phase=phase,
        excluded_decoding_labels=(),
    )


def _build_qwen_profile(phase: str) -> ResumeProfile:
    if phase == SAFE_PHASE:
        return _build_profile(
            profile_name="qwen-safe",
            phase=phase,
            excluded_decoding_labels=(
                "contrastive_penalty_alpha_0.2",
                "contrastive_penalty_alpha_0.8",
            ),
        )
    return _build_profile(
        profile_name="qwen-risky",
        phase=phase,
        excluded_decoding_labels=(),
    )


def _build_mistral_profile(phase: str) -> ResumeProfile:
    if phase == SAFE_PHASE:
        return _build_profile(
            profile_name="mistral-safe",
            phase=phase,
            excluded_decoding_labels=("contrastive_penalty_alpha_0.8",),
        )
    return _build_profile(
        profile_name="mistral-risky",
        phase=phase,
        excluded_decoding_labels=(),
    )


def _build_profile(
    *,
    profile_name: str,
    phase: str,
    excluded_decoding_labels: tuple[str, ...],
) -> ResumeProfile:
    return ResumeProfile(
        profile_name=profile_name,
        phase=phase,
        runtime_mode="docker",
        excluded_decoding_labels=excluded_decoding_labels,
        retry_timeout_failures_on_resume=True,
        retry_oom_failures_on_resume=True,
        retry_infrastructure_failures_on_resume=True,
        retry_structural_failures_on_resume=True,
        generation_timeout_seconds=60,
        startup_timeout_seconds=1800,
        worker_process_mode="ephemeral",
        max_requests_per_worker_process=1,
    )
