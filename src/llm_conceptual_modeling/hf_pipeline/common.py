from __future__ import annotations

from pathlib import Path

from llm_conceptual_modeling.hf_batch.monitoring import status_timestamp_now
from llm_conceptual_modeling.hf_worker.state import (
    mark_worker_ready_for_execution as _worker_state_mark_ready_for_execution,
)


def mark_worker_ready_for_execution(run_dir: Path | None) -> None:
    if run_dir is None:
        return
    _worker_state_mark_ready_for_execution(
        run_dir / "worker_state.json",
        timestamp=status_timestamp_now(),
    )
