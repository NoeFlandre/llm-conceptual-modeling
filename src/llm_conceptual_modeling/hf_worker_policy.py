from __future__ import annotations

_DEFAULT_STARTUP_TIMEOUT_SECONDS = 900.0
_DEFAULT_STAGE_TIMEOUT_SECONDS = 180.0
_DEFAULT_RUN_RETRY_ATTEMPTS = 2


def resolve_startup_timeout_seconds(context_policy: dict[str, object] | None) -> float:
    return _resolve_timeout_seconds(
        context_policy=context_policy,
        field_name="startup_timeout_seconds",
        default_seconds=_DEFAULT_STARTUP_TIMEOUT_SECONDS,
    )


def resolve_stage_timeout_seconds(context_policy: dict[str, object] | None) -> float:
    return _resolve_timeout_seconds(
        context_policy=context_policy,
        field_name="generation_timeout_seconds",
        default_seconds=_DEFAULT_STAGE_TIMEOUT_SECONDS,
    )


def resolve_run_retry_attempts(context_policy: dict[str, object] | None) -> int:
    if context_policy is None:
        return _DEFAULT_RUN_RETRY_ATTEMPTS
    raw_value = context_policy.get("run_retry_attempts", _DEFAULT_RUN_RETRY_ATTEMPTS)
    if isinstance(raw_value, bool):
        raise TypeError("Retry attempts value must be numeric, got bool")
    if isinstance(raw_value, int | float | str):
        attempts = int(float(raw_value))
        return max(1, attempts)
    raise TypeError(f"Retry attempts value must be numeric, got {type(raw_value).__name__}")


def _resolve_timeout_seconds(
    *,
    context_policy: dict[str, object] | None,
    field_name: str,
    default_seconds: float,
) -> float:
    if context_policy is None:
        return default_seconds
    raw_value = context_policy.get(field_name, default_seconds)
    return coerce_timeout_seconds(raw_value)


def coerce_timeout_seconds(raw_value: object) -> float:
    if isinstance(raw_value, bool) or not isinstance(raw_value, int | float | str):
        raise TypeError(f"Timeout value must be numeric, got {type(raw_value).__name__}")
    seconds = float(raw_value)
    if seconds <= 0:
        raise ValueError("Timeout must be positive.")
    return seconds
