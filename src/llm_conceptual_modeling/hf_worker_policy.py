from __future__ import annotations


def resolve_startup_timeout_seconds(context_policy: dict[str, object] | None) -> float:
    if context_policy is None:
        return 900.0
    raw_value = context_policy.get("startup_timeout_seconds", 900)
    return coerce_timeout_seconds(raw_value)


def resolve_stage_timeout_seconds(context_policy: dict[str, object] | None) -> float:
    if context_policy is None:
        return 180.0
    raw_value = context_policy.get("generation_timeout_seconds", 180)
    return coerce_timeout_seconds(raw_value)


def resolve_run_retry_attempts(context_policy: dict[str, object] | None) -> int:
    if context_policy is None:
        return 2
    raw_value = context_policy.get("run_retry_attempts", 2)
    if isinstance(raw_value, bool):
        raise TypeError("Retry attempts value must be numeric, got bool")
    if isinstance(raw_value, int | float | str):
        attempts = int(float(raw_value))
        return max(1, attempts)
    raise TypeError(f"Retry attempts value must be numeric, got {type(raw_value).__name__}")


def coerce_timeout_seconds(raw_value: object) -> float:
    if isinstance(raw_value, bool) or not isinstance(raw_value, int | float | str):
        raise ValueError(f"Unsupported timeout value: {raw_value!r}")
    seconds = float(raw_value)
    if seconds <= 0:
        raise ValueError("Timeout must be positive.")
    return seconds
