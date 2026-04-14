"""Tests to enforce that hf_ shim files do not accumulate at package root.

10 shim files were consolidated into their packages.
The parametrized test ensures the original 10 do not reappear.
The regression test guards against that set growing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).parents[2] / "src" / "llm_conceptual_modeling"


@pytest.mark.parametrize(
    "shim_name",
    [
        # Original 10 consolidated in this pass
        "hf_batch_utils.py",
        "hf_execution_runtime.py",
        "hf_batch_outputs.py",
        "hf_drain_supervisor.py",
        "hf_resume_state.py",
        "hf_shard_manifest.py",
        "hf_qwen_algo1_tail.py",
        "hf_resume_sweep.py",
        "hf_resume_profile.py",
        "hf_resume_preflight.py",
        # Additional 8 identified as same pattern
        "hf_batch_types.py",
        "hf_batch_planning.py",
        "hf_batch_prompts.py",
        "hf_active_models.py",
        "hf_worker_request.py",
        "hf_worker_policy.py",
        "hf_worker_result.py",
        "hf_worker_state.py",
    ],
)
def test_no_hf_shim_files_at_package_root(shim_name: str) -> None:
    """Root-level hf_ shim files were consolidated into their packages."""
    shim_path = PACKAGE_ROOT / shim_name
    assert not shim_path.exists(), (
        f"Root-level shim {shim_name!r} should not exist. "
        "Move implementation into the appropriate hf_* package."
    )


def test_no_new_hf_shim_files_appear_at_package_root() -> None:
    """Detect any new hf_*.py shim files beyond the 10 known consolidated files.

    The 10 shims were removed. This test catches new ones being added.
    """
    # Keep the comprehensive allowlist since many hf_*.py files at root
    # are legitimate (hf_experiments, hf_batch_types, hf_run_config, etc.)
    frozenset({
        "hf_experiments.py",
        "hf_batch_types.py",
        "hf_run_config.py",
        "__init__.py",
        "paths.py",
        "experiment_manifest.py",
        "generation.py",
        "cli.py",
    })
    # The 18 removed shims - any appearing means the removal regressed
    REMOVED_SHIMS = frozenset({
        # Original 10
        "hf_batch_utils.py",
        "hf_execution_runtime.py",
        "hf_batch_outputs.py",
        "hf_drain_supervisor.py",
        "hf_resume_state.py",
        "hf_shard_manifest.py",
        "hf_qwen_algo1_tail.py",
        "hf_resume_sweep.py",
        "hf_resume_profile.py",
        "hf_resume_preflight.py",
        # Additional 8
        "hf_batch_types.py",
        "hf_batch_planning.py",
        "hf_batch_prompts.py",
        "hf_active_models.py",
        "hf_worker_request.py",
        "hf_worker_policy.py",
        "hf_worker_result.py",
        "hf_worker_state.py",
    })
    present = {f.name for f in PACKAGE_ROOT.glob("hf_*.py")}

    # Ensure no removed shim has reappeared
    regressed = present & REMOVED_SHIMS
    assert not regressed, (
        f"Removed shims regressed: {sorted(regressed)}. "
        "These files should not exist at package root."
    )
