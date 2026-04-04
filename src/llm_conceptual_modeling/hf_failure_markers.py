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
)

INFRASTRUCTURE_FAILURE_MESSAGE_MARKERS: tuple[str, ...] = (
    "hf-transformers local inference requires cuda",
    "cuda initialization:",
    "found no nvidia driver",
    "modulenotfounderror",
    "error while finding module specification",
    "no such file or directory",
)

UNSUPPORTED_FAILURE_MESSAGE_MARKERS: tuple[str, ...] = (
    "contrastive search is not supported with stateful models",
    "contrastive search requires `trust_remote_code=true`",
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
