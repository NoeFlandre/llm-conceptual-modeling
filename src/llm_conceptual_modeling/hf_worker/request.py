from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from llm_conceptual_modeling.common.spec_codec import serialize_spec
from llm_conceptual_modeling.hf_batch.types import HFRunSpec


@dataclass(frozen=True)
class HFWorkerRequest:
    request_id: str
    request_json_path: Path
    spec_json_path: Path
    result_json_path: Path
    run_dir: Path


def enqueue_worker_request(
    *,
    queue_dir: Path,
    run_dir: Path,
    spec: HFRunSpec,
    request_id: str,
) -> HFWorkerRequest:
    spec_json_path = run_dir / "worker_spec.json"
    result_json_path = run_dir / "worker_result.json"
    request_json_path = queue_dir / f"{request_id}.request.json"
    spec_json_path.write_text(json.dumps(serialize_spec(spec), indent=2), encoding="utf-8")
    request_json_path.write_text(
        json.dumps(
            {
                "request_id": request_id,
                "spec_json": str(spec_json_path),
                "result_json": str(result_json_path),
                "run_dir": str(run_dir),
                "enqueued_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return HFWorkerRequest(
        request_id=request_id,
        request_json_path=request_json_path,
        spec_json_path=spec_json_path,
        result_json_path=result_json_path,
        run_dir=run_dir,
    )


def load_worker_request(request_json_path: Path) -> HFWorkerRequest:
    payload = json.loads(request_json_path.read_text(encoding="utf-8"))
    return HFWorkerRequest(
        request_id=str(payload.get("request_id", request_json_path.stem.removesuffix(".request"))),
        request_json_path=request_json_path,
        spec_json_path=Path(str(payload["spec_json"])),
        result_json_path=Path(str(payload["result_json"])),
        run_dir=Path(str(payload["run_dir"])),
    )
