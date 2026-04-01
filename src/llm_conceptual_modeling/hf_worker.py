from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from llm_conceptual_modeling.common.hf_transformers import (
    build_runtime_factory,
)
from llm_conceptual_modeling.hf_batch_utils import resolve_hf_token
from llm_conceptual_modeling.hf_spec_codec import deserialize_spec


def _update_worker_state(path: Path, updates: dict[str, object]) -> None:
    payload: dict[str, object] = {}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-json", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)

    spec = deserialize_spec(json.loads(Path(args.spec_json).read_text(encoding="utf-8")))
    run_dir = Path(args.run_dir)
    result_path = Path(args.result_json)
    hf_runtime = build_runtime_factory(hf_token=resolve_hf_token())
    _update_worker_state(
        run_dir / "worker_state.json",
        {
            "worker_pid": os.getpid(),
            "model_loaded": True,
            "model_loaded_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
        },
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
        result_path.write_text(
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

    result_path.write_text(
        json.dumps({"ok": True, "runtime_result": result}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
