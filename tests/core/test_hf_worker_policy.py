from __future__ import annotations

from collections.abc import Iterator, Mapping

import pytest

from llm_conceptual_modeling.hf_worker.policy import (
    coerce_timeout_seconds,
    resolve_run_retry_attempts,
    resolve_stage_timeout_seconds,
    resolve_startup_timeout_seconds,
)


def test_worker_policy_timeout_defaults_match_runtime_contract() -> None:
    assert resolve_startup_timeout_seconds(None) == 900.0
    assert resolve_stage_timeout_seconds(None) == 180.0


def test_worker_policy_timeout_resolution_accepts_numeric_values() -> None:
    context_policy = {
        "startup_timeout_seconds": "120",
        "generation_timeout_seconds": 45,
    }

    assert resolve_startup_timeout_seconds(context_policy) == 120.0
    assert resolve_stage_timeout_seconds(context_policy) == 45.0


def test_worker_policy_accepts_mapping_context_policy() -> None:
    class ContextPolicy(Mapping[str, object]):
        def __init__(self) -> None:
            self._payload = {
                "startup_timeout_seconds": "120",
                "generation_timeout_seconds": 45,
                "run_retry_attempts": 3,
            }

        def get(self, key: str, default: object | None = None) -> object | None:
            return self._payload.get(key, default)

        def __getitem__(self, key: str) -> object:
            return self._payload[key]

        def __iter__(self) -> Iterator[str]:
            return iter(self._payload)

        def __len__(self) -> int:
            return len(self._payload)

    context_policy = ContextPolicy()

    assert resolve_startup_timeout_seconds(context_policy) == 120.0
    assert resolve_stage_timeout_seconds(context_policy) == 45.0
    assert resolve_run_retry_attempts(context_policy) == 3


def test_worker_policy_rejects_boolean_timeouts() -> None:
    with pytest.raises(TypeError, match="Timeout value must be numeric, got bool"):
        coerce_timeout_seconds(True)


def test_worker_policy_retry_attempts_default_and_floor() -> None:
    assert resolve_run_retry_attempts(None) == 2
    assert resolve_run_retry_attempts({"run_retry_attempts": 0}) == 1


def test_worker_policy_retry_attempts_rejects_boolean() -> None:
    with pytest.raises(TypeError, match="numeric"):
        resolve_run_retry_attempts({"run_retry_attempts": True})
