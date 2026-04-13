from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from llm_conceptual_modeling.hf_subprocess import (
    MonitoredCommandTimeout,
    build_hf_download_environment,
    run_monitored_command,
)


def test_run_monitored_command_fails_on_startup_timeout(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-c",
        "import time; time.sleep(10)",
    ]

    with pytest.raises(MonitoredCommandTimeout, match="startup"):
        run_monitored_command(
            command=command,
            run_dir=tmp_path,
            startup_timeout_seconds=0.1,
            stage_timeout_seconds=1.0,
        )


def test_run_monitored_command_fails_on_stale_active_stage(tmp_path: Path) -> None:
    script = """
import json
import sys
import time
from pathlib import Path

run_dir = Path(sys.argv[1])
(run_dir / "active_stage.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
time.sleep(10)
"""
    command = [
        sys.executable,
        "-c",
        script,
        str(tmp_path),
    ]

    with pytest.raises(MonitoredCommandTimeout, match="stage"):
        run_monitored_command(
            command=command,
            run_dir=tmp_path,
            startup_timeout_seconds=1.0,
            stage_timeout_seconds=0.1,
        )

    worker_state = json.loads((tmp_path / "worker_state.json").read_text(encoding="utf-8"))
    assert worker_state["status"] == "failed"
    assert worker_state["failure_phase"] == "stage"


def test_run_monitored_command_records_worker_state_on_success(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-c",
        "print('ok')",
    ]

    completed = run_monitored_command(
        command=command,
        run_dir=tmp_path,
        startup_timeout_seconds=1.0,
        stage_timeout_seconds=1.0,
    )

    worker_state = json.loads((tmp_path / "worker_state.json").read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert worker_state["status"] == "finished"
    assert worker_state["pid"] > 0


def test_build_hf_download_environment_enables_xet_by_default() -> None:
    env = build_hf_download_environment({})

    assert env["HF_HUB_DISABLE_XET"] == "0"
    assert env["HF_HUB_ENABLE_HF_TRANSFER"] == "0"
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_build_hf_download_environment_preserves_pythonpath(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/custom/path")

    env = build_hf_download_environment({"FOO": "bar"})

    assert env.get("PYTHONPATH") == "/custom/path"
