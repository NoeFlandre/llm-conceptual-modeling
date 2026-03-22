import email
from importlib import util
from pathlib import Path
from urllib.error import HTTPError, URLError


def _load_runner_module():
    script_path = Path("scripts/post_revision_debug/run_mistral_probe_matrix.py")
    spec = util.spec_from_file_location("run_mistral_probe_matrix", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load run_mistral_probe_matrix.py")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_probe_specs_cover_the_expected_representative_rows() -> None:
    module = _load_runner_module()

    probe_specs = module._default_probe_specs(
        algo1_rows=[],
        algo2_rows=[],
        algo3_rows=[],
    )

    assert [(spec.algorithm, spec.row_index) for spec in probe_specs] == [
        ("algo1", 0),
        ("algo1", 80),
        ("algo2", 0),
        ("algo2", 160),
        ("algo3", 0),
        ("algo3", 1),
    ]


def test_custom_probe_rows_override_the_defaults() -> None:
    module = _load_runner_module()

    probe_specs = module._default_probe_specs(
        algo1_rows=[3],
        algo2_rows=[7, 11],
        algo3_rows=[13],
    )

    assert [(spec.algorithm, spec.row_index) for spec in probe_specs] == [
        ("algo1", 3),
        ("algo2", 7),
        ("algo2", 11),
        ("algo3", 13),
    ]


def test_call_mistral_with_retry_recovers_from_http_429(monkeypatch) -> None:
    module = _load_runner_module()
    calls: list[str] = []

    class FakeResponse:
        def __init__(self, payload: str) -> None:
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return self.payload.encode("utf-8")

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise HTTPError(
                url=request.full_url,
                code=429,
                msg="Too Many Requests",
                hdrs=email.message.Message(),
                fp=None,
            )
        return FakeResponse('{"choices":[{"message":{"content":"{\\"edges\\":[]}"}}]}')

    sleep_calls: list[float] = []

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    actual = module._call_mistral_with_retry(
        api_key="test-key",
        model="mistral-small-2603",
        prompt="prompt text",
        logger=None,
    )

    assert len(calls) == 2
    assert sleep_calls == [1.0]
    assert actual["choices"][0]["message"]["content"] == '{"edges":[]}'


def test_call_mistral_with_retry_recovers_from_urlerror(monkeypatch) -> None:
    module = _load_runner_module()
    calls: list[str] = []

    class FakeResponse:
        def __init__(self, payload: str) -> None:
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return self.payload.encode("utf-8")

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise URLError("temporary network issue")
        return FakeResponse('{"choices":[{"message":{"content":"{\\"edges\\":[]}"}}]}')

    sleep_calls: list[float] = []

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    actual = module._call_mistral_with_retry(
        api_key="test-key",
        model="mistral-small-2603",
        prompt="prompt text",
        logger=None,
    )

    assert len(calls) == 2
    assert sleep_calls == [1.0]
    assert actual["choices"][0]["message"]["content"] == '{"edges":[]}'


def test_call_mistral_with_retry_exponential_backoff_for_urlerror(monkeypatch) -> None:
    module = _load_runner_module()
    calls: list[str] = []

    class FakeResponse:
        def __init__(self, payload: str) -> None:
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return self.payload.encode("utf-8")

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        if len(calls) < 3:
            raise URLError("temporary network issue")
        return FakeResponse('{"choices":[{"message":{"content":"{\\"edges\\":[]}"}}]}')

    sleep_calls: list[float] = []

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    actual = module._call_mistral_with_retry(
        api_key="test-key",
        model="mistral-small-2603",
        prompt="prompt text",
        logger=None,
    )

    assert len(calls) == 3
    assert sleep_calls == [1.0, 2.0]
    assert actual["choices"][0]["message"]["content"] == '{"edges":[]}'


def test_load_existing_response_respects_resume_flag(tmp_path) -> None:
    module = _load_runner_module()
    response_path = tmp_path / "response.json"
    response_path.write_text('{"choices":[{"message":{"content":"cached"}}]}')

    cached = module._load_existing_response(
        response_path=response_path,
        resume=True,
    )
    missing = module._load_existing_response(
        response_path=response_path,
        resume=False,
    )

    assert cached["choices"][0]["message"]["content"] == "cached"
    assert missing is None


def test_build_model_failure_record_preserves_context() -> None:
    module = _load_runner_module()

    actual = module._build_model_failure_record(
        algorithm="algo2",
        row_index=7,
        model="mistral-small-2603",
        historical_model="gpt-5",
        error=RuntimeError("temporary failure"),
    )

    assert actual == {
        "algorithm": "algo2",
        "row_index": 7,
        "model": "mistral-small-2603",
        "historical_model": "gpt-5",
        "error_type": "RuntimeError",
        "error_message": "temporary failure",
    }
