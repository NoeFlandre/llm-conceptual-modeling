from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_conceptual_modeling.hf_worker.result import (
    load_runtime_result,
    raise_missing_result_artifact,
)


def test_load_runtime_result_returns_runtime_payload(tmp_path: Path) -> None:
    result_json_path = tmp_path / "worker_result.json"
    result_json_path.write_text(
        json.dumps(
            {
                "ok": True,
                "runtime_result": {"raw_row": {}, "runtime": {}, "raw_response": "{}"},
            }
        ),
        encoding="utf-8",
    )

    result = load_runtime_result(result_json_path)

    assert result["raw_response"] == "{}"


def test_load_runtime_result_raises_runtime_error_for_failure_payload(tmp_path: Path) -> None:
    result_json_path = tmp_path / "worker_result.json"
    result_json_path.write_text(
        json.dumps({"ok": False, "error": {"type": "ValueError", "message": "bad schema"}}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="ValueError: bad schema"):
        load_runtime_result(result_json_path)


def test_raise_missing_result_artifact_includes_context() -> None:
    with pytest.raises(RuntimeError, match="stdout='out' stderr='err'"):
        raise_missing_result_artifact(
            context="HF worker subprocess",
            stdout="out",
            stderr="err",
        )
