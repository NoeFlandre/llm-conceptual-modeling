from __future__ import annotations

STRUCTURAL_FAILURE_MESSAGE_MARKERS: tuple[str, ...] = (
    "model did not return valid structured output:",
    "invalid edge item shape:",
    "unsupported structured response shape for schema",
    "structured edge_list response must contain a list of edges",
    "structured edge_list flat string response must contain an even number of items",
    "structurally invalid algo",
    "structured response returned an empty",
    "structured response returned a null",
    "jsondecodeerror",
    "could not parse tuple content:",
    "unsupported cache type:",
    "contrastive search requires dynamic cache",
    "contrastive search requires `trust_remote_code=true`",
    "'>' not supported between instances of 'nonetype' and 'int'",
)

INFRASTRUCTURE_FAILURE_MESSAGE_MARKERS: tuple[str, ...] = (
    "hf-transformers local inference requires cuda",
    "cuda initialization:",
    "found no nvidia driver",
    "modulenotfounderror",
    "error while finding module specification",
    "no such file or directory",
    "exited before writing a result artifact",
)

UNSUPPORTED_FAILURE_MESSAGE_MARKERS: tuple[str, ...] = (
    "contrastive search is not supported with stateful models",
)

RETRYABLE_WORKER_FAILURE_MESSAGE_MARKERS: tuple[str, ...] = (
    *STRUCTURAL_FAILURE_MESSAGE_MARKERS,
    *INFRASTRUCTURE_FAILURE_MESSAGE_MARKERS,
    "out of memory",
)


def message_contains_any(message: str, markers: tuple[str, ...]) -> bool:
    lowered_message = message.lower()
    return any(marker in lowered_message for marker in markers)


def is_retryable_worker_failure_message(message: str) -> bool:
    return "brokenpipeerror:" in message.lower() or message_contains_any(
        message,
        RETRYABLE_WORKER_FAILURE_MESSAGE_MARKERS,
    )


def classify_failure(*, error_type: str, message: str) -> str:
    if error_type == "MonitoredCommandTimeout" or "MonitoredCommandTimeout" in message:
        return "timeout"
    if message_contains_any(message, INFRASTRUCTURE_FAILURE_MESSAGE_MARKERS):
        return "infrastructure"
    if "out of memory" in message.lower():
        return "oom"
    if message_contains_any(message, UNSUPPORTED_FAILURE_MESSAGE_MARKERS):
        return "unsupported"
    if error_type == "StaleRunState":
        return "stale"
    if error_type == "JSONDecodeError":
        return "structural"
    if error_type in {"ValueError", "RuntimeError"} and message_contains_any(
        message,
        STRUCTURAL_FAILURE_MESSAGE_MARKERS,
    ):
        return "structural"
    return "other"


def is_retryable_runtime_failure(*, error_type: str, message: str) -> bool:
    if "brokenpipeerror:" in message.lower():
        return True
    failure_kind = classify_failure(error_type=error_type, message=message)
    return failure_kind in {"timeout", "oom", "infrastructure", "structural"}
