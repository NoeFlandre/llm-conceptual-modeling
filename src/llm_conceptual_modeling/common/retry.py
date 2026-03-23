from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar
from urllib.error import HTTPError, URLError

import httpx

try:
    try:
        from mistralai.models.sdkerror import SDKError
    except ImportError:  # pragma: no cover - compatibility with older SDKs
        from mistralai.client.errors.sdkerror import SDKError
    try:
        from mistralai.utils.retries import PermanentError
    except ImportError:  # pragma: no cover - compatibility with older SDKs
        from mistralai.client.utils.retries import PermanentError
except ImportError:  # pragma: no cover - exercised indirectly in import-light tests

    class SDKError(Exception):
        pass

    class PermanentError(Exception):
        inner: Exception | None = None


T = TypeVar("T")

logger = logging.getLogger(__name__)


def call_with_retry(
    *,
    operation: Callable[[], T],
    operation_name: str,
    max_attempts: int = 4,
    initial_delay_seconds: float = 0.5,
    backoff_factor: float = 2.0,
    max_delay_seconds: float = 8.0,
    retry_http_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504),
    sleep_fn: Callable[[float], None] = time.sleep,
) -> T:
    attempt = 1
    delay_seconds = initial_delay_seconds

    while True:
        try:
            return operation()
        except HTTPError as error:
            retryable = error.code in retry_http_status_codes
            exception = error
        except URLError as error:
            retryable = True
            exception = error
        except httpx.HTTPError as error:
            retryable = True
            exception = error
        except PermanentError as error:
            if isinstance(error.inner, (URLError, httpx.HTTPError)):
                retryable = True
                exception = error.inner
            else:
                retryable = False
                exception = error
        except SDKError as error:
            status_code = getattr(error, "status_code", None)
            if status_code is None:
                raw_response = getattr(error, "raw_response", None)
                status_code = getattr(raw_response, "status_code", None)
            retryable = status_code in retry_http_status_codes
            exception = error

        if not retryable or attempt >= max_attempts:
            logger.error(
                "Transient transport error exhausted retries; "
                "operation=%s attempts=%s/%s error_type=%s",
                operation_name,
                attempt,
                max_attempts,
                type(exception).__name__,
            )
            raise exception

        next_attempt = attempt + 1
        logger.warning(
            "Transient transport error; retrying operation=%s attempt=%s/%s delay_seconds=%.3f",
            operation_name,
            next_attempt,
            max_attempts,
            delay_seconds,
        )
        sleep_fn(delay_seconds)
        attempt = next_attempt
        delay_seconds = min(delay_seconds * backoff_factor, max_delay_seconds)
