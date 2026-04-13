from __future__ import annotations

from llm_conceptual_modeling.common.failure_markers import (
    classify_failure,
    is_retryable_runtime_failure,
)


def test_classify_failure_detects_missing_result_artifact_as_infrastructure() -> None:
    assert (
        classify_failure(
            error_type="RuntimeError",
            message="Persistent HF worker exited before writing a result artifact.",
        )
        == "infrastructure"
    )


def test_is_retryable_runtime_failure_retries_stage_timeout() -> None:
    assert (
        is_retryable_runtime_failure(
            error_type="MonitoredCommandTimeout",
            message="Monitored command exceeded stage timeout of 20.0 seconds.",
        )
        is True
    )


def test_is_retryable_runtime_failure_does_not_retry_unsupported_failure() -> None:
    assert (
        is_retryable_runtime_failure(
            error_type="RuntimeError",
            message=(
                "ValueError: contrastive search is not supported with stateful models, "
                "such as Qwen3_5ForCausalLM"
            ),
        )
        is False
    )


def test_is_retryable_runtime_failure_retries_known_mistral_contrastive_type_error() -> None:
    assert (
        is_retryable_runtime_failure(
            error_type="RuntimeError",
            message="TypeError: '>' not supported between instances of 'NoneType' and 'int'",
        )
        is True
    )


def test_is_retryable_runtime_failure_retries_contrastive_generate_trust_remote_code_error() -> None:
    assert (
        is_retryable_runtime_failure(
            error_type="RuntimeError",
            message=(
                "ValueError: Contrastive Search requires `trust_remote_code=True` in your "
                "`generate` call, since it loads https://hf.co/transformers-community/contrastive-search."
            ),
        )
        is True
    )


def test_hf_failure_markers_public_api_lives_in_common_module() -> None:
    assert classify_failure.__module__ == "llm_conceptual_modeling.common.failure_markers"
    assert is_retryable_runtime_failure.__module__ == "llm_conceptual_modeling.common.failure_markers"
