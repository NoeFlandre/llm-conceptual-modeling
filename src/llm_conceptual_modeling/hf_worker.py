from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm_conceptual_modeling.common.hf_transformers import (
    _torch,
    build_runtime_factory,
)
from llm_conceptual_modeling.hf_batch_utils import resolve_hf_token
from llm_conceptual_modeling.hf_spec_codec import deserialize_spec
from llm_conceptual_modeling.hf_worker_request import load_worker_request
from llm_conceptual_modeling.hf_worker_state import mark_worker_prefetching_model


def serve_request_queue(
    *,
    queue_dir: Path,
    max_requests: int | None = None,
    idle_sleep_seconds: float = 0.1,
) -> int:
    hf_runtime = build_runtime_factory(hf_token=resolve_hf_token())
    served_count = 0
    while max_requests is None or served_count < max_requests:
        request_path = _claim_next_request(queue_dir)
        if request_path is None:
            if max_requests is None:
                time.sleep(idle_sleep_seconds)
                continue
            break
        request = load_worker_request(request_path)
        _execute_request(
            spec_json_path=request.spec_json_path,
            result_json_path=request.result_json_path,
            run_dir=request.run_dir,
            hf_runtime=hf_runtime,
            requests_served_by_process=served_count + 1,
        )
        _release_runtime_cache()
        request_path.unlink(missing_ok=True)
        served_count += 1
    return served_count


def _claim_next_request(queue_dir: Path) -> Path | None:
    for request_path in sorted(queue_dir.glob("*.request.json")):
        claimed_path = request_path.with_suffix("").with_suffix(".claimed.json")
        try:
            request_path.rename(claimed_path)
        except FileNotFoundError:
            continue
        return claimed_path
    return None


def _execute_request(
    *,
    spec_json_path: Path,
    result_json_path: Path,
    run_dir: Path,
    hf_runtime: Any,
    requests_served_by_process: int = 1,
) -> int:
    spec = deserialize_spec(json.loads(spec_json_path.read_text(encoding="utf-8")))
    mark_worker_prefetching_model(
        run_dir / "worker_state.json",
        worker_pid=os.getpid(),
        requests_served_by_process=requests_served_by_process,
        timestamp=datetime.now(UTC).isoformat(),
    )
    from llm_conceptual_modeling.hf_experiments import _run_algo1, _run_algo2, _run_algo3

    try:
        if spec.algorithm == "algo1":
            result = _run_algo1(spec, hf_runtime=hf_runtime, run_dir=run_dir)
        elif spec.algorithm == "algo2":
            result = _run_algo2(spec, hf_runtime=hf_runtime, run_dir=run_dir)
        elif spec.algorithm == "algo3":
            result = _run_algo3(spec, hf_runtime=hf_runtime, run_dir=run_dir)
        else:
            raise ValueError(f"Unsupported algorithm: {spec.algorithm}")
    except Exception as error:
        result_json_path.write_text(
            json.dumps(
                {
                    "ok": False,
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return 1

    result_json_path.write_text(
        json.dumps({"ok": True, "runtime_result": result}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


def _release_runtime_cache() -> None:
    try:
        torch = _torch()
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-json")
    parser.add_argument("--result-json")
    parser.add_argument("--run-dir")
    parser.add_argument("--queue-dir")
    parser.add_argument("--max-requests", type=int)
    args = parser.parse_args(argv)

    if args.queue_dir:
        return serve_request_queue(
            queue_dir=Path(args.queue_dir),
            max_requests=args.max_requests,
        )
    if not args.spec_json or not args.result_json or not args.run_dir:
        raise SystemExit("--spec-json, --result-json, and --run-dir are required")

    hf_runtime = build_runtime_factory(hf_token=resolve_hf_token())
    return _execute_request(
        spec_json_path=Path(args.spec_json),
        result_json_path=Path(args.result_json),
        run_dir=Path(args.run_dir),
        hf_runtime=hf_runtime,
    )


if __name__ == "__main__":
    raise SystemExit(main())
