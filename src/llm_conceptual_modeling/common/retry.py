from __future__ import annotations

import importlib
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar, cast
from urllib.error import HTTPError, URLError

import httpx


class SDKError(Exception):
    pass


class PermanentError(Exception):
    inner: Exception | None = None

    def __init__(self, inner: Exception | None = None) -> None:
        self.inner = inner
        super().__init__(str(inner) if inner is not None else "")


def _load_symbol(module_name: str, symbol_name: str) -> Any | None:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    return getattr(module, symbol_name, None)


_sdk_error_type = _load_symbol("mistralai.models.sdkerror", "SDKError") or _load_symbol(
    "mistralai.client.errors.sdkerror", "SDKError"
)
_permanent_error_type = _load_symbol("mistralai.utils.retries", "PermanentError") or _load_symbol(
    "mistralai.client.utils.retries", "PermanentError"
)


T = TypeVar("T")

logger = logging.getLogger(__name__)


def _status_code_from_sdk_error(error: Exception) -> int | None:
    status_code = getattr(error, "status_code", None)
    if status_code is not None:
        return cast(int | None, status_code)
    raw_response = getattr(error, "raw_response", None)
    return cast(int | None, getattr(raw_response, "status_code", None))


def _retryable_exception_from_wrapped_error(
    error: Exception,
    *,
    retry_http_status_codes: tuple[int, ...],
) -> tuple[bool, Exception]:
    inner = getattr(error, "inner", None)
    if isinstance(inner, HTTPError):
        return inner.code in retry_http_status_codes, inner
    if isinstance(inner, (URLError, httpx.HTTPError)):
        return True, cast(Exception, inner)
    return False, error


def _retryable_sdk_error(
    error: Exception,
    *,
    retry_http_status_codes: tuple[int, ...],
) -> bool:
    return _status_code_from_sdk_error(error) in retry_http_status_codes


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
        except Exception as error:
            if _permanent_error_type is not None and isinstance(error, _permanent_error_type):
                retryable, exception = _retryable_exception_from_wrapped_error(
                    error,
                    retry_http_status_codes=retry_http_status_codes,
                )
            elif isinstance(error, PermanentError):
                retryable, exception = _retryable_exception_from_wrapped_error(
                    error,
                    retry_http_status_codes=retry_http_status_codes,
                )
            elif _sdk_error_type is not None and isinstance(error, _sdk_error_type):
                retryable = _retryable_sdk_error(
                    error,
                    retry_http_status_codes=retry_http_status_codes,
                )
                exception = error
            elif isinstance(error, SDKError):
                retryable = _retryable_sdk_error(
                    error,
                    retry_http_status_codes=retry_http_status_codes,
                )
                exception = error
            else:
                raise

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
