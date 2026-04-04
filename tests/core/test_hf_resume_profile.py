from llm_conceptual_modeling.hf_resume_profile import resolve_resume_profile


def test_resolve_resume_profile_defaults_to_docker_and_safe_profiles() -> None:
    profile = resolve_resume_profile("hf-paper-batch-algo1-olmo-current")

    assert profile.runtime_mode == "docker"
    assert profile.profile_name == "olmo-safe"
    assert profile.excluded_decoding_labels == ("contrastive_penalty_alpha_0.8",)


def test_resolve_resume_profile_keeps_qwen_coverage_by_default() -> None:
    profile = resolve_resume_profile("hf-paper-batch-algo2-qwen-current")

    assert profile.runtime_mode == "docker"
    assert profile.profile_name == "qwen-safe"
    assert profile.excluded_decoding_labels == ()


def test_resolve_resume_profile_supports_explicit_full_coverage_override() -> None:
    profile = resolve_resume_profile(
        "hf-paper-batch-algo1-olmo-current",
        full_coverage=True,
    )

    assert profile.profile_name == "full-coverage"
    assert profile.excluded_decoding_labels == ()
