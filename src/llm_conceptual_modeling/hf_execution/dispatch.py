"""Execution dispatch helpers.

Pure functions that select the right runtime path for a single run spec and
map an HFTransformersRuntimeFactory to the generic RuntimeFactory interface.
No subprocess logic, no state mutation beyond what the caller controls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from llm_conceptual_modeling.common.hf_transformers import HFTransformersRuntimeFactory
from llm_conceptual_modeling.hf_batch.types import HFRunSpec, RuntimeFactory, RuntimeResult
from llm_conceptual_modeling.hf_pipeline.algo1 import run_algo1 as _run_algo1
from llm_conceptual_modeling.hf_pipeline.algo2 import run_algo2 as _run_algo2
from llm_conceptual_modeling.hf_pipeline.algo3 import run_algo3 as _run_algo3


def execute_run(
    *,
    spec: HFRunSpec,
    runtime_factory: RuntimeFactory,
    dry_run: bool,
    run_dir: Path | None = None,
) -> RuntimeResult:
    """Execute a single run spec, returning a RuntimeResult.

    When dry_run is True, returns a synthetic result without calling the
    runtime_factory. When dry_run is False, delegates to runtime_factory.
    """
    if dry_run:
        return {
            "raw_row": dict(spec.raw_context),
            "runtime": {
                "thinking_mode_supported": spec.runtime_profile.supports_thinking_toggle,
                "device": spec.runtime_profile.device,
                "dtype": spec.runtime_profile.dtype,
                "quantization": spec.runtime_profile.quantization,
            },
            "raw_response": "[]",
        }
    try:
        runtime_callable = cast(Any, runtime_factory)
        return runtime_callable(spec, run_dir=run_dir)
    except TypeError as error:
        if "run_dir" not in str(error):
            raise
        return runtime_factory(spec)


def runtime_factory_from_hf_runtime(
    hf_runtime: HFTransformersRuntimeFactory,
) -> RuntimeFactory:
    """Return a RuntimeFactory that dispatches by algorithm using hf_runtime."""

    def runtime(spec: HFRunSpec, *, run_dir: Path | None = None) -> RuntimeResult:
        if spec.algorithm == "algo1":
            return _run_algo1(spec, hf_runtime=hf_runtime, run_dir=run_dir)
        if spec.algorithm == "algo2":
            return _run_algo2(spec, hf_runtime=hf_runtime, run_dir=run_dir)
        if spec.algorithm == "algo3":
            return _run_algo3(spec, hf_runtime=hf_runtime, run_dir=run_dir)
        raise ValueError(f"Unsupported algorithm: {spec.algorithm}")

    return runtime
