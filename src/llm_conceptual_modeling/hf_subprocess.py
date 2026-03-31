from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class MonitoredCommandTimeout(RuntimeError):
    phase: str
    timeout_seconds: float

    def __str__(self) -> str:
        return f"Monitored command exceeded {self.phase} timeout of {self.timeout_seconds} seconds."


def build_hf_download_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(base or os.environ)
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    return env


def run_monitored_command(
    *,
    command: list[str],
    run_dir: Path,
    startup_timeout_seconds: float,
    stage_timeout_seconds: float,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(env) if env is not None else None,
    )
    started_at = time.monotonic()
    active_stage_path = run_dir / "active_stage.json"

    while process.poll() is None:
        now = time.monotonic()
        if active_stage_path.exists():
            stage_age_seconds = time.time() - active_stage_path.stat().st_mtime
            if stage_age_seconds > stage_timeout_seconds:
                _terminate_process(process)
                raise MonitoredCommandTimeout("stage", stage_timeout_seconds)
        elif now - started_at > startup_timeout_seconds:
            _terminate_process(process)
            raise MonitoredCommandTimeout("startup", startup_timeout_seconds)
        time.sleep(0.05)

    stdout, stderr = process.communicate()
    return subprocess.CompletedProcess(
        args=command,
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=2)
        return
    except subprocess.TimeoutExpired:
        pass
    process.kill()
    process.wait(timeout=2)
