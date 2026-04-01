from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HUGGING_FACE_BUCKET_REPO = "NoeFlandre/llm-variability-conceptual-modeling"
HUGGING_FACE_BUCKET_URL = f"https://huggingface.co/{HUGGING_FACE_BUCKET_REPO}"
GITHUB_REPO_URL = "https://github.com/NoeFlandre/llm-conceptual-modeling"


def default_results_root() -> str:
    configured_root = os.environ.get("LCM_RESULTS_ROOT")
    if configured_root:
        return str(Path(configured_root).expanduser())
    return str(REPO_ROOT / "data" / "results")


def default_analysis_artifacts_root() -> str:
    configured_root = os.environ.get("LCM_ANALYSIS_ARTIFACTS_ROOT")
    if configured_root:
        return str(Path(configured_root).expanduser())
    return str(REPO_ROOT / "data" / "analysis_artifacts")


def default_inputs_root() -> str:
    configured_root = os.environ.get("LCM_INPUTS_ROOT")
    if configured_root:
        return str(Path(configured_root).expanduser())
    return str(REPO_ROOT / "data" / "inputs")


def default_revision_tracker_root() -> str:
    return str(Path(default_analysis_artifacts_root()) / "revision_tracker")
