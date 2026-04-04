from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_conceptual_modeling.hf_batch_types import HFRunSpec, RuntimeResult
from llm_conceptual_modeling.hf_failure_markers import (
    is_retryable_runtime_failure,
)
from llm_conceptual_modeling.hf_subprocess import (
    MonitoredCommandTimeout,
    _terminate_process,
    build_hf_download_environment,
)
from llm_conceptual_modeling.hf_worker_policy import (
    resolve_run_retry_attempts as _resolve_run_retry_attempts,
)
from llm_conceptual_modeling.hf_worker_policy import (
    resolve_stage_timeout_seconds as _resolve_stage_timeout_seconds,
)
from llm_conceptual_modeling.hf_worker_policy import (
    resolve_startup_timeout_seconds as _resolve_startup_timeout_seconds,
)
from llm_conceptual_modeling.hf_worker_request import enqueue_worker_request
from llm_conceptual_modeling.hf_worker_result import load_runtime_result
from llm_conceptual_modeling.hf_worker_state import (
    read_worker_state,
    worker_has_started_stage_execution,
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
        startup_timeout_seconds = _resolve_startup_timeout_seconds(spec.context_policy)
        stage_timeout_seconds = _resolve_stage_timeout_seconds(spec.context_policy)
        retry_attempts = _resolve_run_retry_attempts(spec.context_policy)
        for attempt in range(1, retry_attempts + 1):
            self._recycle_process_if_budget_exhausted()
            process = self._ensure_process()
            request_path, result_json_path = self._enqueue_request(spec=spec, run_dir=run_dir)
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
            except Exception as error:
                self._terminate_process(process)
                self._process = None
                self._requests_served_by_process = 0
                if attempt < retry_attempts and _is_retryable_runtime_error(error):
                    continue
                raise
            finally:
                request_path.unlink(missing_ok=True)
        raise RuntimeError("Persistent HF worker retry loop exhausted without a terminal result.")

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
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            env=build_hf_download_environment(self.env),
        )

    def _enqueue_request(self, *, spec: HFRunSpec, run_dir: Path) -> tuple[Path, Path]:
        self._next_request_id += 1
        request_id = f"{self._next_request_id:08d}"
        request = enqueue_worker_request(
            queue_dir=self.queue_dir,
            run_dir=run_dir,
            spec=spec,
            request_id=request_id,
        )
        return request.request_json_path, request.result_json_path

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
                request_path.unlink(missing_ok=True)
                return load_runtime_result(result_json_path)
            if process.poll() is not None:
                raise RuntimeError("Persistent HF worker exited before writing a result artifact.")
            if active_stage_path.exists():
                stage_age_seconds = time.time() - active_stage_path.stat().st_mtime
                if stage_age_seconds > stage_timeout_seconds:
                    raise MonitoredCommandTimeout("stage", stage_timeout_seconds)
            elif worker_state_path.exists():
                worker_state = read_worker_state(worker_state_path)
                heartbeat_age_seconds = time.time() - worker_state_path.stat().st_mtime
                if worker_has_started_stage_execution(worker_state):
                    if heartbeat_age_seconds > stage_timeout_seconds:
                        raise MonitoredCommandTimeout("stage", stage_timeout_seconds)
                elif time.monotonic() - started_at > startup_timeout_seconds:
                    raise MonitoredCommandTimeout("startup", startup_timeout_seconds)
            elif time.monotonic() - started_at > startup_timeout_seconds:
                raise MonitoredCommandTimeout("startup", startup_timeout_seconds)
            time.sleep(0.05)

    def _terminate_process(self, process: subprocess.Popen[str] | Any) -> None:
        _terminate_process(process)

def _is_retryable_runtime_error(error: Exception) -> bool:
    return is_retryable_runtime_failure(
        error_type=type(error).__name__,
        message=str(error),
    )
