from __future__ import annotations

import math

import pandas as pd

from llm_conceptual_modeling.common.csv_schema import assert_required_columns
from llm_conceptual_modeling.common.types import PathLike

_REQUIRED_COLUMNS = ("metric", "n", "mean", "sample_std")


def write_replication_budget_analysis(
    input_csv_paths: list[PathLike] | tuple[PathLike, ...],
    output_csv_path: PathLike,
    *,
    relative_half_width_target: float = 0.05,
    z_score: float = 1.96,
) -> None:
    if relative_half_width_target <= 0:
        raise ValueError("relative_half_width_target must be positive.")
    if z_score <= 0:
        raise ValueError("z_score must be positive.")

    output_frames: list[pd.DataFrame] = []
    for input_csv_path in input_csv_paths:
        frame = pd.read_csv(input_csv_path)
        assert_required_columns(frame, list(_REQUIRED_COLUMNS), label="replication-budget columns")
        enriched = frame.copy()
        enriched["relative_half_width_target"] = float(relative_half_width_target)
        enriched["z_score"] = float(z_score)
        enriched["precision_margin"] = enriched["mean"].abs() * float(relative_half_width_target)
        enriched["required_total_runs"] = enriched.apply(
            lambda row: _required_total_runs(
                observed_runs=int(row["n"]),
                mean=float(row["mean"]),
                sample_std=float(row["sample_std"]),
                relative_half_width_target=float(relative_half_width_target),
                z_score=float(z_score),
            ),
            axis=1,
        )
        enriched["additional_runs_needed"] = enriched["required_total_runs"] - enriched["n"]
        enriched["requirement_status"] = enriched.apply(_requirement_status, axis=1)
        output_frames.append(enriched)

    pd.concat(output_frames, ignore_index=True).to_csv(output_csv_path, index=False)


def required_total_runs_from_row(
    *,
    observed_runs: int,
    mean: float,
    sample_std: float,
    relative_half_width_target: float = 0.05,
    z_score: float = 1.96,
) -> int:
    return _required_total_runs(
        observed_runs=observed_runs,
        mean=mean,
        sample_std=sample_std,
        relative_half_width_target=relative_half_width_target,
        z_score=z_score,
    )


def _required_total_runs(
    *,
    observed_runs: int,
    mean: float,
    sample_std: float,
    relative_half_width_target: float,
    z_score: float,
) -> int:
    if observed_runs <= 0:
        raise ValueError("observed_runs must be positive.")
    if not math.isfinite(sample_std):
        return observed_runs
    if sample_std < 0:
        raise ValueError("sample_std must be non-negative.")
    if mean == 0 or sample_std == 0:
        return observed_runs

    precision_margin = abs(mean) * relative_half_width_target
    required = math.ceil(((z_score * sample_std) / precision_margin) ** 2)
    return max(required, observed_runs)


def _requirement_status(row: pd.Series) -> str:
    if row["additional_runs_needed"] > 0:
        return "needs_more_runs"
    if row["mean"] == 0:
        return "satisfied_zero_mean"
    return "satisfied"
