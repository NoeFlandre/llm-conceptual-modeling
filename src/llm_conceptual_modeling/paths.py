"""Compatibility shim for repository path helpers."""

from llm_conceptual_modeling.common.paths import (
    GITHUB_REPO_URL,
    HUGGING_FACE_BUCKET_REPO,
    HUGGING_FACE_BUCKET_URL,
    REPO_ROOT,
    default_analysis_artifacts_root,
    default_inputs_root,
    default_results_root,
    default_revision_tracker_root,
)

__all__ = [
    "GITHUB_REPO_URL",
    "HUGGING_FACE_BUCKET_REPO",
    "HUGGING_FACE_BUCKET_URL",
    "REPO_ROOT",
    "default_analysis_artifacts_root",
    "default_inputs_root",
    "default_results_root",
    "default_revision_tracker_root",
]
