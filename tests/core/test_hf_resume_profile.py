from llm_conceptual_modeling.hf_resume.profile import resolve_resume_profile


def test_resolve_resume_profile_defaults_to_docker_and_safe_profiles() -> None:
    profile = resolve_resume_profile("hf-paper-batch-algo1-olmo-current")

    assert profile.runtime_mode == "docker"
    assert profile.profile_name == "olmo-safe"
    assert profile.phase == "safe"
    assert profile.excluded_decoding_labels == ("contrastive_penalty_alpha_0.8",)
    assert profile.retry_timeout_failures_on_resume is True
    assert profile.retry_structural_failures_on_resume is True
    assert profile.generation_timeout_seconds == 60
    assert profile.startup_timeout_seconds == 1800


def test_resolve_resume_profile_keeps_qwen_coverage_by_default() -> None:
    profile = resolve_resume_profile("hf-paper-batch-algo2-qwen-current")

    assert profile.runtime_mode == "docker"
    assert profile.profile_name == "qwen-safe"
    assert profile.phase == "safe"
    assert profile.excluded_decoding_labels == (
        "contrastive_penalty_alpha_0.2",
        "contrastive_penalty_alpha_0.8",
    )


def test_resolve_resume_profile_supports_explicit_risky_phase() -> None:
    profile = resolve_resume_profile(
        "hf-paper-batch-algo2-qwen-current",
        phase="risky",
    )

    assert profile.profile_name == "qwen-risky"
    assert profile.phase == "risky"
    assert profile.excluded_decoding_labels == ()
    assert profile.retry_oom_failures_on_resume is True


def test_resolve_resume_profile_supports_explicit_full_coverage_override() -> None:
    profile = resolve_resume_profile(
        "hf-paper-batch-algo1-olmo-current",
        full_coverage=True,
    )

    assert profile.profile_name == "full-coverage"
    assert profile.phase == "risky"
    assert profile.excluded_decoding_labels == ()


def test_resume_profile_safe_phase_excludes_contrastive_decoding_per_family() -> None:
    olmo_safe = resolve_resume_profile("hf-paper-batch-algo1-olmo-current", phase="safe")
    assert olmo_safe.excluded_decoding_labels == ("contrastive_penalty_alpha_0.8",)

    qwen_safe = resolve_resume_profile("hf-paper-batch-algo2-qwen-current", phase="safe")
    assert qwen_safe.excluded_decoding_labels == (
        "contrastive_penalty_alpha_0.2",
        "contrastive_penalty_alpha_0.8",
    )

    mistral_safe = resolve_resume_profile(
        "hf-paper-batch-algo1-mistral-current", phase="safe"
    )
    assert mistral_safe.excluded_decoding_labels == ("contrastive_penalty_alpha_0.8",)


def test_resume_profile_risky_phase_includes_all_decoding() -> None:
    qwen_risky = resolve_resume_profile(
        "hf-paper-batch-algo2-qwen-current",
        phase="risky",
    )
    assert qwen_risky.excluded_decoding_labels == ()
    assert qwen_risky.profile_name == "qwen-risky"
