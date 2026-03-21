from urllib.error import HTTPError, URLError

from llm_conceptual_modeling.common.retry import call_with_retry


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
            hdrs=None,
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
