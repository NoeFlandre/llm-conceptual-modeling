from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.common.csv_schema import assert_required_columns
from llm_conceptual_modeling.common.types import PathLike

FACTOR_COLUMNS = ("Example", "Counter-Example", "Number of Words", "Depth")
PAIR_COLUMNS = ("Source Subgraph Name", "Target Subgraph Name")
DEFAULT_METRIC_COLUMN = "Recall"
_MANIFEST_COLUMNS = ("model", "input_path")
_REPETITION_COLUMN = "Repetition"
_SOURCE_COLUMN = "source_input"
_GROUP_COLUMNS = (*FACTOR_COLUMNS, *PAIR_COLUMNS)

_CONDITION_COLUMNS = (
    "source_input",
    "model",
    *FACTOR_COLUMNS,
    *PAIR_COLUMNS,
    "metric",
    "observed_runs",
    "mean",
    "sample_std",
    "min",
    "max",
    "range_width",
    "coefficient_of_variation",
    "required_total_runs",
    "additional_runs_needed",
    "requirement_status",
)

_CUMULATIVE_COLUMNS = (
    "source_input",
    "model",
    *FACTOR_COLUMNS,
    *PAIR_COLUMNS,
    "metric",
    "prefix_runs",
    "cumulative_mean",
)

_SUMMARY_COLUMNS = (
    "model",
    "metric",
    "condition_count",
    "observed_runs",
    "zero_mean_conditions",
    "nonzero_mean_conditions",
    "varying_conditions",
    "varying_condition_share",
    "varying_nonzero_conditions",
    "mean_cv_nonzero",
    "median_cv_nonzero",
    "max_cv_nonzero",
    "conditions_requiring_more_runs",
    "median_required_total_runs",
    "max_required_total_runs",
)


@dataclass(frozen=True)
class ReplicationStabilityResult:
    condition_budget: pd.DataFrame
    cumulative_means: pd.DataFrame
    model_summary: pd.DataFrame


def estimate_required_runs(
    *,
    observed_runs: int,
    mean: float,
    sample_std: float | None,
    relative_half_width_target: float,
    z_score: float,
) -> int | None:
    """Estimate the total runs needed for a relative confidence half-width."""
    _validate_precision_parameters(relative_half_width_target, z_score)
    _validate_observation_inputs(observed_runs, mean, sample_std)
    if sample_std is None or not math.isfinite(sample_std):
        return None
    if mean == 0:
        return None
    if sample_std == 0:
        return observed_runs

    precision_margin = abs(mean) * relative_half_width_target
    estimate = math.ceil(((z_score * sample_std) / precision_margin) ** 2)
    return max(estimate, observed_runs)


def analyze_replication_stability(
    manifest_path: PathLike,
    output_dir: PathLike,
    *,
    metric_column: str = DEFAULT_METRIC_COLUMN,
    relative_half_width_target: float = 0.05,
    z_score: float = 1.96,
) -> ReplicationStabilityResult:
    """Analyze exact Algorithm 3 conditions and write the result tables."""
    _validate_precision_parameters(relative_half_width_target, z_score)
    frames = _read_manifest_frames(manifest_path, metric_column=metric_column)
    condition_budget = _condition_budget(
        frames,
        metric_column=metric_column,
        relative_half_width_target=relative_half_width_target,
        z_score=z_score,
    )
    cumulative_means = _cumulative_means(frames, metric_column=metric_column)
    model_summary = _model_summary(condition_budget, metric=metric_column)
    result = ReplicationStabilityResult(
        condition_budget=condition_budget,
        cumulative_means=cumulative_means,
        model_summary=model_summary,
    )
    _write_result(result, output_dir)
    return result


def _validate_precision_parameters(relative_half_width_target: float, z_score: float) -> None:
    if relative_half_width_target <= 0:
        raise ValueError("relative_half_width_target must be positive.")
    if z_score <= 0:
        raise ValueError("z_score must be positive.")


def _validate_observation_inputs(
    observed_runs: int,
    mean: float,
    sample_std: float | None,
) -> None:
    if observed_runs <= 0:
        raise ValueError("observed_runs must be positive.")
    if not math.isfinite(mean):
        raise ValueError("mean must be finite.")
    _validate_sample_std(sample_std)


def _validate_sample_std(sample_std: float | None) -> None:
    if sample_std is None:
        return
    if not math.isfinite(sample_std):
        return
    if sample_std < 0:
        raise ValueError("sample_std must be non-negative.")


def _read_manifest_frames(
    manifest_path: PathLike,
    *,
    metric_column: str,
) -> list[pd.DataFrame]:
    manifest_path = Path(manifest_path)
    manifest = pd.read_csv(manifest_path)
    assert_required_columns(manifest, list(_MANIFEST_COLUMNS), label="manifest columns")
    if manifest.empty:
        raise ValueError("The replication-stability manifest is empty.")

    frames: list[pd.DataFrame] = []
    model_index = manifest.columns.get_loc("model")
    input_index = manifest.columns.get_loc("input_path")
    for row in manifest.to_numpy():
        model = _manifest_value(row[model_index], "model")
        input_text = _manifest_value(row[input_index], "input_path")
        input_path = Path(input_text)
        if not input_path.is_absolute():
            input_path = manifest_path.parent / input_path
        if not input_path.is_file():
            raise ValueError(f"Manifest input does not exist: {input_path}")
        frame = pd.read_csv(input_path)
        frames.append(
            _prepare_input_frame(
                frame,
                model=model,
                source_input=input_text,
                metric_column=metric_column,
            )
        )
    combined = pd.concat(frames, ignore_index=True)
    _reject_duplicate_repetitions(combined)
    return [combined]


def _manifest_value(value: object, label: str) -> str:
    if pd.isna(value):
        raise ValueError(f"Manifest {label} values must not be missing.")
    text = str(value).strip()
    if not text:
        raise ValueError(f"Manifest {label} values must not be empty.")
    return text


def _prepare_input_frame(
    frame: pd.DataFrame,
    *,
    model: str,
    source_input: str,
    metric_column: str,
) -> pd.DataFrame:
    required = [_REPETITION_COLUMN, *_GROUP_COLUMNS, metric_column]
    assert_required_columns(frame, required, label="replication-stability input columns")
    _validate_required_values(frame, required, source_input)

    prepared = frame[required].copy()
    repetition = pd.to_numeric(prepared[_REPETITION_COLUMN], errors="coerce")
    metric = pd.to_numeric(prepared[metric_column], errors="coerce")
    _validate_repetition_values(repetition, source_input)
    _validate_metric_values(metric, source_input)

    prepared[_REPETITION_COLUMN] = repetition.astype(int)
    prepared[metric_column] = metric
    prepared["model"] = model
    prepared[_SOURCE_COLUMN] = source_input
    return prepared


def _validate_required_values(
    frame: pd.DataFrame,
    required: list[str],
    source_input: str,
) -> None:
    if frame[required].isna().any().any():
        raise ValueError(
            "Replication-stability input contains missing required values: "
            f"{source_input}"
        )


def _validate_repetition_values(repetition: pd.Series, source_input: str) -> None:
    if repetition.isna().any() or (repetition % 1 != 0).any():
        raise ValueError(f"Repetition values must be integer-valued: {source_input}")


def _validate_metric_values(metric: pd.Series, source_input: str) -> None:
    if metric.isna().any() or not metric.map(math.isfinite).all():
        raise ValueError(f"Metric values must be finite: {source_input}")


def _reject_duplicate_repetitions(frame: pd.DataFrame) -> None:
    duplicate_columns = ["model", *_GROUP_COLUMNS, _REPETITION_COLUMN]
    if frame.duplicated(duplicate_columns).any():
        raise ValueError("Each exact condition must contain at most one row per repetition.")


def _condition_budget(
    frames: list[pd.DataFrame],
    *,
    metric_column: str,
    relative_half_width_target: float,
    z_score: float,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for frame in frames:
        group_columns = ["model", *_GROUP_COLUMNS]
        for group_key, group in frame.groupby(group_columns):
            records.append(
                _condition_record(
                    group_key=group_key,
                    group=group.sort_values(_REPETITION_COLUMN),
                    metric_column=metric_column,
                    relative_half_width_target=relative_half_width_target,
                    z_score=z_score,
                )
            )
    return pd.DataFrame.from_records(records, columns=_CONDITION_COLUMNS)


def _condition_record(
    *,
    group_key: object,
    group: pd.DataFrame,
    metric_column: str,
    relative_half_width_target: float,
    z_score: float,
) -> dict[str, object]:
    model, *factor_values = _as_tuple(group_key, len(_GROUP_COLUMNS) + 1)
    values = group[metric_column]
    observed_runs = int(values.size)
    mean = float(values.mean())
    sample_std = _sample_std(values)
    required = estimate_required_runs(
        observed_runs=observed_runs,
        mean=mean,
        sample_std=sample_std,
        relative_half_width_target=relative_half_width_target,
        z_score=z_score,
    )
    minimum = float(values.min())
    maximum = float(values.max())
    range_width = maximum - minimum
    return {
        "source_input": _source_label(group[_SOURCE_COLUMN]),
        "model": model,
        **_factor_mapping(factor_values),
        "metric": metric_column,
        "observed_runs": observed_runs,
        "mean": mean,
        "sample_std": sample_std,
        "min": minimum,
        "max": maximum,
        "range_width": range_width,
        "coefficient_of_variation": _coefficient_of_variation(mean, sample_std),
        "required_total_runs": required,
        "additional_runs_needed": None if required is None else required - observed_runs,
        "requirement_status": _requirement_status(
            observed_runs=observed_runs,
            mean=mean,
            sample_std=sample_std,
            required=required,
        ),
    }


def _as_tuple(value: object, expected_size: int) -> tuple[object, ...]:
    values = value if isinstance(value, tuple) else (value,)
    if len(values) != expected_size:
        raise ValueError(f"Expected {expected_size} grouped values, got {len(values)}.")
    return values


def _factor_mapping(factor_values: list[object]) -> dict[str, object]:
    return {
        column: factor_values[index]
        for index, column in enumerate(_GROUP_COLUMNS)
    }


def _sample_std(values: pd.Series) -> float | None:
    if len(values) < 2:
        return None
    return float(values.std(ddof=1))


def _coefficient_of_variation(mean: float, sample_std: float | None) -> float | None:
    if sample_std is None or mean == 0:
        return None
    return sample_std / abs(mean)


def _requirement_status(
    *,
    observed_runs: int,
    mean: float,
    sample_std: float | None,
    required: int | None,
) -> str:
    if observed_runs < 2 or sample_std is None:
        return "insufficient_observations"
    if mean == 0:
        return "zero_mean_not_estimable"
    if _requires_more_runs(required, observed_runs):
        return "requires_more_runs"
    return "stable_at_observed_runs"


def _requires_more_runs(required: int | None, observed_runs: int) -> bool:
    return required is not None and required > observed_runs


def _source_label(values: pd.Series) -> str:
    return "|".join(sorted({str(value) for value in values}))


def _cumulative_means(frames: list[pd.DataFrame], *, metric_column: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for frame in frames:
        group_columns = ["model", *_GROUP_COLUMNS]
        for group_key, group in frame.groupby(group_columns):
            model, *factor_values = _as_tuple(group_key, len(_GROUP_COLUMNS) + 1)
            ordered = group.sort_values(_REPETITION_COLUMN)
            for prefix_runs in range(1, len(ordered) + 1):
                record = {
                    "source_input": _source_label(ordered[_SOURCE_COLUMN]),
                    "model": model,
                    **_factor_mapping(factor_values),
                    "metric": metric_column,
                    "prefix_runs": prefix_runs,
                    "cumulative_mean": float(ordered[metric_column].iloc[:prefix_runs].mean()),
                }
                records.append(record)
    return pd.DataFrame.from_records(records, columns=_CUMULATIVE_COLUMNS)


def _model_summary(condition_budget: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for model, group in condition_budget.groupby("model"):
        nonzero = group[group["mean"] != 0]
        varying = group[group["range_width"] > 0]
        varying_nonzero = varying[varying["mean"] != 0]
        cv = nonzero["coefficient_of_variation"].dropna()
        required = group["required_total_runs"].dropna()
        records.append(
            {
                "model": model,
                "metric": metric,
                "condition_count": int(len(group)),
                "observed_runs": int(group["observed_runs"].max()),
                "zero_mean_conditions": int(len(group) - len(nonzero)),
                "nonzero_mean_conditions": int(len(nonzero)),
                "varying_conditions": int(len(varying)),
                "varying_condition_share": float(len(varying) / len(group)),
                "varying_nonzero_conditions": int(len(varying_nonzero)),
                "mean_cv_nonzero": _series_statistic(cv, "mean"),
                "median_cv_nonzero": _series_statistic(cv, "median"),
                "max_cv_nonzero": _series_statistic(cv, "max"),
                "conditions_requiring_more_runs": int(
                    (group["requirement_status"] == "requires_more_runs").sum()
                ),
                "median_required_total_runs": _series_statistic(required, "median"),
                "max_required_total_runs": _series_statistic(required, "max"),
            }
        )
    return pd.DataFrame.from_records(records, columns=_SUMMARY_COLUMNS)


def _series_statistic(series: pd.Series, statistic: str) -> float | None:
    if series.empty:
        return None
    value = getattr(series, statistic)()
    return float(value)


def _write_result(result: ReplicationStabilityResult, output_dir: PathLike) -> None:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    result.condition_budget.to_csv(
        output_dir_path / "condition_budget.csv",
        index=False,
    )
    result.cumulative_means.to_csv(
        output_dir_path / "cumulative_means.csv",
        index=False,
    )
    result.model_summary.to_csv(
        output_dir_path / "model_summary.csv",
        index=False,
    )
