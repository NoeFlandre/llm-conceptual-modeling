from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from llm_conceptual_modeling.hf_batch_types import HFRunSpec, RuntimeResult
from llm_conceptual_modeling.hf_spec_codec import serialize_spec
from llm_conceptual_modeling.hf_subprocess import (
    MonitoredCommandTimeout,
    _terminate_process,
    build_hf_download_environment,
)


@dataclass
class PersistentHFWorkerSession:
    queue_dir: Path
    worker_python: str = sys.executable
    env: dict[str, str] | None = None
    max_requests_per_process: int | None = None

    def __post_init__(self) -> None:
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self._process: subprocess.Popen[str] | Any | None = None
        self._next_request_id = 0
        self._requests_served_by_process = 0

    def run_spec(self, *, spec: HFRunSpec, run_dir: Path) -> RuntimeResult:
        run_dir.mkdir(parents=True, exist_ok=True)
        self._recycle_process_if_budget_exhausted()
        process = self._ensure_process()
        request_path, result_json_path = self._enqueue_request(spec=spec, run_dir=run_dir)
        startup_timeout_seconds = _resolve_startup_timeout_seconds(spec.context_policy)
        stage_timeout_seconds = _resolve_stage_timeout_seconds(spec.context_policy)
        try:
            result = self._wait_for_result(
                process=process,
                request_path=request_path,
                result_json_path=result_json_path,
                run_dir=run_dir,
                startup_timeout_seconds=startup_timeout_seconds,
                stage_timeout_seconds=stage_timeout_seconds,
            )
            self._requests_served_by_process += 1
            return result
        except Exception:
            self._terminate_process(process)
            self._process = None
            self._requests_served_by_process = 0
            raise
        finally:
            request_path.unlink(missing_ok=True)

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        self._terminate_process(process)
        self._process = None
        self._requests_served_by_process = 0

    def _ensure_process(self) -> subprocess.Popen[str] | Any:
        process = self._process
        if process is not None and process.poll() is None:
            return process
        self._process = self._spawn_worker_process()
        self._requests_served_by_process = 0
        return self._process

    def _recycle_process_if_budget_exhausted(self) -> None:
        if self.max_requests_per_process is None:
            return
        process = self._process
        if process is None or process.poll() is not None:
            return
        if self._requests_served_by_process < self.max_requests_per_process:
            return
        self._terminate_process(process)
        self._process = None
        self._requests_served_by_process = 0

    def _spawn_worker_process(self) -> subprocess.Popen[str]:
        return subprocess.Popen(
            [
                self.worker_python,
                "-m",
                "llm_conceptual_modeling.hf_worker",
                "--queue-dir",
                str(self.queue_dir),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=build_hf_download_environment(self.env),
        )

    def _enqueue_request(self, *, spec: HFRunSpec, run_dir: Path) -> tuple[Path, Path]:
        self._next_request_id += 1
        request_id = f"{self._next_request_id:08d}"
        spec_json_path = run_dir / "worker_spec.json"
        result_json_path = run_dir / "worker_result.json"
        request_path = self.queue_dir / f"{request_id}.request.json"
        spec_json_path.write_text(json.dumps(serialize_spec(spec), indent=2), encoding="utf-8")
        request_path.write_text(
            json.dumps(
                {
                    "request_id": request_id,
                    "spec_json": str(spec_json_path),
                    "result_json": str(result_json_path),
                    "run_dir": str(run_dir),
                    "enqueued_at": _timestamp_now(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return request_path, result_json_path

    def _wait_for_result(
        self,
        *,
        process: subprocess.Popen[str] | Any,
        request_path: Path,
        result_json_path: Path,
        run_dir: Path,
        startup_timeout_seconds: float,
        stage_timeout_seconds: float,
    ) -> RuntimeResult:
        started_at = time.monotonic()
        active_stage_path = run_dir / "active_stage.json"
        worker_state_path = run_dir / "worker_state.json"
        while True:
            if result_json_path.exists():
                worker_payload = json.loads(result_json_path.read_text(encoding="utf-8"))
                request_path.unlink(missing_ok=True)
                if worker_payload.get("ok"):
                    return cast(RuntimeResult, worker_payload["runtime_result"])
                error = cast(dict[str, str], worker_payload["error"])
                raise RuntimeError(f"{error['type']}: {error['message']}")
            if process.poll() is not None:
                raise RuntimeError("Persistent HF worker exited before writing a result artifact.")
            if active_stage_path.exists():
                stage_age_seconds = time.time() - active_stage_path.stat().st_mtime
                if stage_age_seconds > stage_timeout_seconds:
                    raise MonitoredCommandTimeout("stage", stage_timeout_seconds)
            elif worker_state_path.exists():
                heartbeat_age_seconds = time.time() - worker_state_path.stat().st_mtime
                if heartbeat_age_seconds > stage_timeout_seconds:
                    raise MonitoredCommandTimeout("stage", stage_timeout_seconds)
            elif time.monotonic() - started_at > startup_timeout_seconds:
                raise MonitoredCommandTimeout("startup", startup_timeout_seconds)
            time.sleep(0.05)

    def _terminate_process(self, process: subprocess.Popen[str] | Any) -> None:
        _terminate_process(process)


def _resolve_startup_timeout_seconds(context_policy: dict[str, object] | None) -> float:
    if context_policy is None:
        return 900.0
    raw_value = context_policy.get("startup_timeout_seconds", 900)
    return _coerce_timeout_seconds(raw_value)


def _resolve_stage_timeout_seconds(context_policy: dict[str, object] | None) -> float:
    if context_policy is None:
        return 180.0
    raw_value = context_policy.get("generation_timeout_seconds", 180)
    return _coerce_timeout_seconds(raw_value)


def _coerce_timeout_seconds(raw_value: object) -> float:
    if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
        raise ValueError(f"Unsupported timeout value: {raw_value!r}")
    seconds = float(raw_value)
    if seconds <= 0:
        raise ValueError("Timeout must be positive.")
    return seconds


def _timestamp_now() -> str:
    return datetime.now(UTC).isoformat()
