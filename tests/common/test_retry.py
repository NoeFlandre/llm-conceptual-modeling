import email
from urllib.error import HTTPError, URLError

import httpx

from llm_conceptual_modeling.common.retry import PermanentError, SDKError, call_with_retry


def test_call_with_retry_retries_transient_urle_error_before_succeeding() -> None:
    calls = {"count": 0}
    delays: list[float] = []

    def operation() -> str:
        calls["count"] += 1
        if calls["count"] < 3:
            raise URLError("temporary network issue")
        return "ok"

    actual = call_with_retry(
        operation=operation,
        operation_name="test operation",
        max_attempts=4,
        initial_delay_seconds=0.1,
        sleep_fn=delays.append,
    )

    assert actual == "ok"
    assert calls["count"] == 3
    assert delays == [0.1, 0.2]


def test_call_with_retry_does_not_retry_nonretryable_http_error() -> None:
    calls = {"count": 0}
    delays: list[float] = []

    def operation() -> str:
        calls["count"] += 1
        raise HTTPError(
            url="https://example.com",
            code=400,
            msg="bad request",
            hdrs=email.message.Message(),
            fp=None,
        )

    try:
        call_with_retry(
            operation=operation,
            operation_name="test operation",
            max_attempts=4,
            sleep_fn=delays.append,
        )
    except HTTPError as error:
        assert error.code == 400
    else:
        raise AssertionError("Expected HTTPError to be raised")

    assert calls["count"] == 1
    assert delays == []


def test_call_with_retry_retries_httpx_transport_errors_before_succeeding() -> None:
    calls = {"count": 0}
    delays: list[float] = []

    def operation() -> str:
        calls["count"] += 1
        if calls["count"] < 3:
            raise httpx.ConnectError("temporary network issue")
        return "ok"

    actual = call_with_retry(
        operation=operation,
        operation_name="test operation",
        max_attempts=4,
        initial_delay_seconds=0.1,
        sleep_fn=delays.append,
    )

    assert actual == "ok"
    assert calls["count"] == 3
    assert delays == [0.1, 0.2]


def test_call_with_retry_retries_sdk_permanent_error_wrapping_transport_failure() -> None:
    calls = {"count": 0}
    delays: list[float] = []

    def operation() -> str:
        calls["count"] += 1
        if calls["count"] < 3:
            raise PermanentError(httpx.ConnectError("temporary network issue"))
        return "ok"

    actual = call_with_retry(
        operation=operation,
        operation_name="test operation",
        max_attempts=4,
        initial_delay_seconds=0.1,
        sleep_fn=delays.append,
    )

    assert actual == "ok"
    assert calls["count"] == 3
    assert delays == [0.1, 0.2]


def test_call_with_retry_does_not_retry_sdk_permanent_error_wrapping_nonretryable_http_error() -> None:
    calls = {"count": 0}
    delays: list[float] = []

    def operation() -> str:
        calls["count"] += 1
        raise PermanentError(
            HTTPError(
                url="https://example.com",
                code=400,
                msg="bad request",
                hdrs=email.message.Message(),
                fp=None,
            )
        )

    try:
        call_with_retry(
            operation=operation,
            operation_name="test operation",
            max_attempts=4,
            initial_delay_seconds=0.1,
            sleep_fn=delays.append,
        )
    except HTTPError as error:
        assert error.code == 400
    else:
        raise AssertionError("Expected HTTPError to be raised")

    assert calls["count"] == 1
    assert delays == []


def test_call_with_retry_retries_sdk_error_429_before_succeeding() -> None:
    calls = {"count": 0}
    delays: list[float] = []

    class RateLimitedSDKError(SDKError):
        def __init__(self) -> None:
            super().__init__("rate limited")
            self.status_code = 429

    def operation() -> str:
        calls["count"] += 1
        if calls["count"] < 3:
            raise RateLimitedSDKError()
        return "ok"

    actual = call_with_retry(
        operation=operation,
        operation_name="test operation",
        max_attempts=4,
        initial_delay_seconds=0.1,
        sleep_fn=delays.append,
    )

    assert actual == "ok"
    assert calls["count"] == 3
    assert delays == [0.1, 0.2]
