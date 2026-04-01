from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
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
    worker_state_path = run_dir / "worker_state.json"
    _write_worker_state(
        worker_state_path,
        {
            "status": "running",
            "pid": process.pid,
            "started_at": _timestamp_now(),
            "updated_at": _timestamp_now(),
        },
    )

    while process.poll() is None:
        now = time.monotonic()
        if active_stage_path.exists():
            stage_age_seconds = time.time() - active_stage_path.stat().st_mtime
            _write_worker_state(
                worker_state_path,
                {
                    "status": "running",
                    "pid": process.pid,
                    "last_stage_heartbeat_at": _timestamp_now(),
                    "stage_age_seconds": round(stage_age_seconds, 3),
                    "updated_at": _timestamp_now(),
                },
            )
            if stage_age_seconds > stage_timeout_seconds:
                _terminate_process(process)
                _write_worker_state(
                    worker_state_path,
                    {
                        "status": "failed",
                        "pid": process.pid,
                        "failure_type": "MonitoredCommandTimeout",
                        "failure_phase": "stage",
                        "finished_at": _timestamp_now(),
                        "updated_at": _timestamp_now(),
                    },
                )
                raise MonitoredCommandTimeout("stage", stage_timeout_seconds)
        elif now - started_at > startup_timeout_seconds:
            _terminate_process(process)
            _write_worker_state(
                worker_state_path,
                {
                    "status": "failed",
                    "pid": process.pid,
                    "failure_type": "MonitoredCommandTimeout",
                    "failure_phase": "startup",
                    "finished_at": _timestamp_now(),
                    "updated_at": _timestamp_now(),
                },
            )
            raise MonitoredCommandTimeout("startup", startup_timeout_seconds)
        time.sleep(0.05)

    stdout, stderr = process.communicate()
    _write_worker_state(
        worker_state_path,
        {
            "status": "finished" if process.returncode == 0 else "failed",
            "pid": process.pid,
            "returncode": process.returncode,
            "finished_at": _timestamp_now(),
            "updated_at": _timestamp_now(),
        },
    )
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


def _write_worker_state(path: Path, updates: dict[str, object]) -> None:
    payload: dict[str, object] = {}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _timestamp_now() -> str:
    return datetime.now(UTC).isoformat()
