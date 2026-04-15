"""Path parsing helpers for analysis bundle writers."""
from __future__ import annotations

from pathlib import Path

from llm_conceptual_modeling.paths import default_results_root


def _path_triplet(path: Path) -> tuple[str, str, str]:
    parts = path.parts
    aggregated_index = parts.index("aggregated")
    return (
        parts[aggregated_index + 1],
        parts[aggregated_index + 2],
        parts[aggregated_index + 3],
    )


def _discover_main_results_root(results_root: Path) -> Path:
    candidates = [
        results_root.parent / "results",
        results_root.parent.parent / "results",
        Path(default_results_root()),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(default_results_root())
