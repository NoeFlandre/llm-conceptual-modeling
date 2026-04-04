from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResumeProfile:
    profile_name: str
    runtime_mode: str
    excluded_decoding_labels: tuple[str, ...]


def resolve_resume_profile(
    results_root_name: str,
    *,
    full_coverage: bool = False,
) -> ResumeProfile:
    if full_coverage:
        return ResumeProfile(
            profile_name="full-coverage",
            runtime_mode="docker",
            excluded_decoding_labels=(),
        )

    normalized_name = results_root_name.lower()
    if "olmo" in normalized_name:
        return ResumeProfile(
            profile_name="olmo-safe",
            runtime_mode="docker",
            excluded_decoding_labels=("contrastive_penalty_alpha_0.8",),
        )
    if "qwen" in normalized_name:
        return ResumeProfile(
            profile_name="qwen-safe",
            runtime_mode="docker",
            excluded_decoding_labels=(),
        )
    if "mistral" in normalized_name:
        return ResumeProfile(
            profile_name="mistral-safe",
            runtime_mode="docker",
            excluded_decoding_labels=("contrastive_penalty_alpha_0.8",),
        )
    return ResumeProfile(
        profile_name="default",
        runtime_mode="docker",
        excluded_decoding_labels=(),
    )
