from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from llm_conceptual_modeling.analysis.replication_budget import required_total_runs
from llm_conceptual_modeling.common.types import PathLike

_DEFAULT_MODELS = (
    "Qwen/Qwen3.5-9B",
    "mistralai/Ministral-3-8B-Instruct-2512",
)


@dataclass(frozen=True)
class ReplicationBudgetProfile:
    name: str
    confidence_level: float
    z_score: float
    relative_half_width_target: float


_DEFAULT_PROFILES = (
    ReplicationBudgetProfile(
        name="ci95_rel05",
        confidence_level=0.95,
        z_score=1.96,
        relative_half_width_target=0.05,
    ),
    ReplicationBudgetProfile(
        name="ci90_rel05",
        confidence_level=0.90,
        z_score=1.645,
        relative_half_width_target=0.05,
    ),
)

_GROUPINGS = (
    ("overall", ()),
    ("algorithm", ("algorithm",)),
    ("graph_source", ("graph_source",)),
    ("model", ("model",)),
    ("algorithm_model", ("algorithm", "model")),
    ("algorithm_model_graph_source", ("algorithm", "model", "graph_source")),
    ("algorithm_model_decoding", ("algorithm", "model", "decoding_condition")),
    (
        "algorithm_model_graph_source_decoding",
        ("algorithm", "model", "graph_source", "decoding_condition"),
    ),
    (
        "algorithm_model_decoding_metric",
        ("algorithm", "model", "decoding_condition", "metric"),
    ),
    (
        "algorithm_model_graph_source_decoding_metric",
        ("algorithm", "model", "graph_source", "decoding_condition", "metric"),
    ),
)

_MODEL_COLUMN_PREFIXES = {
    "Qwen/Qwen3.5-9B": "qwen",
    "mistralai/Ministral-3-8B-Instruct-2512": "mistral",
}

_ALGORITHM_LABELS = {
    "algo1": "Algorithm 1",
    "algo2": "Algorithm 2",
    "algo3": "Algorithm 3",
}

_ALGORITHM_ORDER = {
    "algo1": 1,
    "algo2": 2,
    "algo3": 3,
}

_DECODING_ORDER = {
    "greedy": 1,
    "beam_num_beams_2": 2,
    "beam_num_beams_6": 3,
    "contrastive_penalty_alpha_0.2": 4,
    "contrastive_penalty_alpha_0.8": 5,
}


def write_replication_budget_sufficiency_summary(
    *,
    results_root: PathLike,
    output_csv_path: PathLike,
    models: tuple[str, ...] | None = None,
    expected_replications: int = 5,
    profiles: tuple[ReplicationBudgetProfile, ...] = _DEFAULT_PROFILES,
) -> None:
    if expected_replications <= 0:
        raise ValueError("expected_replications must be positive.")

    results_root_path = Path(results_root)
    ledger = _read_ledger(results_root_path / "ledger.json")
    selected_models = set(models or _DEFAULT_MODELS)
    observations = _ledger_metric_observations(ledger=ledger, models=selected_models)
    if observations.empty:
        raise ValueError("No finished ledger metric observations matched the requested models.")

    budget_rows = _condition_metric_budget_rows(
        observations=observations,
        profiles=profiles,
        expected_replications=expected_replications,
    )
    summary = _aggregate_budget_rows(
        budget_rows=budget_rows,
        observations=observations,
        profiles=profiles,
    )
    summary.to_csv(output_csv_path, index=False)


def write_compact_replication_budget_sufficiency_table(
    *,
    results_root: PathLike,
    output_csv_path: PathLike,
    models: tuple[str, ...] | None = None,
    expected_replications: int = 5,
    profiles: tuple[ReplicationBudgetProfile, ...] = _DEFAULT_PROFILES,
    include_graph_source: bool = False,
) -> None:
    if expected_replications <= 0:
        raise ValueError("expected_replications must be positive.")

    results_root_path = Path(results_root)
    ledger = _read_ledger(results_root_path / "ledger.json")
    selected_models = set(models or _DEFAULT_MODELS)
    observations = _ledger_metric_observations(ledger=ledger, models=selected_models)
    if observations.empty:
        raise ValueError("No finished ledger metric observations matched the requested models.")

    budget_rows = _condition_metric_budget_rows(
        observations=observations,
        profiles=profiles,
        expected_replications=expected_replications,
    )
    compact = _compact_budget_table(
        budget_rows=budget_rows,
        observations=observations,
        models=tuple(model for model in _DEFAULT_MODELS if model in selected_models),
        profiles=profiles,
        include_graph_source=include_graph_source,
    )
    compact.to_csv(output_csv_path, index=False)


def _read_ledger(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ledger_metric_observations(*, ledger: dict[str, Any], models: set[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in ledger.get("records", []):
        if not isinstance(record, dict) or record.get("status") != "finished":
            continue
        identity = record.get("identity")
        winner = record.get("winner")
        if not isinstance(identity, dict) or not isinstance(winner, dict):
            continue
        model = str(identity.get("model", ""))
        if model not in models:
            continue
        metrics = winner.get("metrics")
        if not isinstance(metrics, dict):
            continue
        run_key = _run_key(identity)
        for metric, raw_value in metrics.items():
            if not isinstance(raw_value, (int, float)) or not math.isfinite(float(raw_value)):
                continue
            rows.append(
                {
                    "run_key": run_key,
                    "algorithm": str(identity["algorithm"]),
                    "graph_source": str(identity.get("graph_source", "default")),
                    "model": model,
                    "decoding_condition": str(identity["condition_label"]),
                    "pair_name": str(identity["pair_name"]),
                    "condition_bits": str(identity["condition_bits"]),
                    "replication": int(identity["replication"]),
                    "metric": str(metric),
                    "value": float(raw_value) * 100.0,
                }
            )
    return pd.DataFrame.from_records(rows)


def _run_key(identity: dict[str, object]) -> str:
    return "|".join(
        [
            str(identity["algorithm"]),
            str(identity["model"]),
            str(identity["condition_label"]),
            str(identity.get("graph_source", "default")),
            str(identity["pair_name"]),
            str(identity["condition_bits"]),
            str(identity["replication"]),
        ]
    )


def _condition_metric_budget_rows(
    *,
    observations: pd.DataFrame,
    profiles: tuple[ReplicationBudgetProfile, ...],
    expected_replications: int,
) -> pd.DataFrame:
    group_columns = [
        "algorithm",
        "graph_source",
        "model",
        "decoding_condition",
        "pair_name",
        "condition_bits",
        "metric",
    ]
    condition_rows: list[dict[str, object]] = []
    for group_key, group in observations.groupby(group_columns, sort=True, dropna=False):
        key_values = _group_key_tuple(group_key, expected_size=len(group_columns))
        algorithm, graph_source, model, decoding_condition, pair_name, condition_bits, metric = (
            key_values
        )
        observed_runs = int(group["replication"].nunique())
        mean = float(group["value"].mean())
        sample_std = float(group["value"].std(ddof=1))
        for profile in profiles:
            required = required_total_runs(
                observed_runs=observed_runs,
                mean=mean,
                sample_std=sample_std,
                relative_half_width_target=profile.relative_half_width_target,
                z_score=profile.z_score,
            )
            additional_runs_needed = max(required - observed_runs, 0)
            condition_rows.append(
                {
                    "profile": profile.name,
                    "confidence_level": profile.confidence_level,
                    "z_score": profile.z_score,
                    "relative_half_width_target": profile.relative_half_width_target,
                    "algorithm": algorithm,
                    "graph_source": graph_source,
                    "model": model,
                    "decoding_condition": decoding_condition,
                    "pair_name": pair_name,
                    "condition_bits": condition_bits,
                    "metric": metric,
                    "observed_runs": observed_runs,
                    "expected_replications": expected_replications,
                    "mean": mean,
                    "sample_std": sample_std,
                    "required_total_runs": required,
                    "additional_runs_needed": additional_runs_needed,
                    "needs_more_runs": additional_runs_needed > 0,
                    "below_expected_replications": observed_runs < expected_replications,
                }
            )
    return pd.DataFrame.from_records(condition_rows)


def _aggregate_budget_rows(
    *,
    budget_rows: pd.DataFrame,
    observations: pd.DataFrame,
    profiles: tuple[ReplicationBudgetProfile, ...],
) -> pd.DataFrame:
    output_rows: list[dict[str, object]] = []
    source_finished_run_count = int(observations["run_key"].nunique())
    for profile in profiles:
        profile_budget = budget_rows[budget_rows["profile"] == profile.name]
        for grouping_name, grouping_columns in _GROUPINGS:
            if grouping_columns:
                grouped_budget = profile_budget.groupby(
                    list(grouping_columns),
                    sort=True,
                    dropna=False,
                )
                for group_key, group in grouped_budget:
                    key_values = _group_key_values(grouping_columns, group_key)
                    output_rows.append(
                        _aggregate_group(
                            profile=profile,
                            grouping_name=grouping_name,
                            group=group,
                            key_values=key_values,
                            source_finished_run_count=source_finished_run_count,
                        )
                    )
            else:
                output_rows.append(
                    _aggregate_group(
                        profile=profile,
                        grouping_name=grouping_name,
                        group=profile_budget,
                        key_values={},
                        source_finished_run_count=source_finished_run_count,
                    )
                )
        underpowered = profile_budget[profile_budget["needs_more_runs"]].sort_values(
            [
                "algorithm",
                "graph_source",
                "model",
                "decoding_condition",
                "metric",
                "pair_name",
                "condition_bits",
            ],
            kind="mergesort",
        )
        for _, row in underpowered.iterrows():
            output_rows.append(
                _aggregate_group(
                    profile=profile,
                    grouping_name="underpowered_condition_metric",
                    group=pd.DataFrame.from_records([row.to_dict()]),
                    key_values={
                        "algorithm": row["algorithm"],
                        "graph_source": row["graph_source"],
                        "model": row["model"],
                        "decoding_condition": row["decoding_condition"],
                        "metric": row["metric"],
                        "pair_name": row["pair_name"],
                        "condition_bits": row["condition_bits"],
                    },
                    source_finished_run_count=source_finished_run_count,
                )
            )
    summary = pd.DataFrame.from_records(output_rows)
    sort_columns = [
        "profile",
        "grouping",
        "algorithm",
        "model",
        "decoding_condition",
        "metric",
        "pair_name",
        "condition_bits",
    ]
    return summary.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def _group_key_values(columns: tuple[str, ...], key: object) -> dict[str, object]:
    values = _group_key_tuple(key, expected_size=len(columns))
    return dict(zip(columns, values, strict=True))


def _group_key_tuple(key: object, *, expected_size: int) -> tuple[object, ...]:
    if expected_size == 1:
        value = key[0] if isinstance(key, tuple) else key
        return (value,)
    if not isinstance(key, tuple) or len(key) != expected_size:
        raise ValueError(f"Expected grouped key with {expected_size} values, got {key!r}.")
    return key


def _aggregate_group(
    *,
    profile: ReplicationBudgetProfile,
    grouping_name: str,
    group: pd.DataFrame,
    key_values: dict[str, object],
    source_finished_run_count: int,
) -> dict[str, object]:
    condition_count = int(len(group))
    needs_more = int(group["needs_more_runs"].sum())
    below_expected = int(group["below_expected_replications"].sum())
    return {
        "profile": profile.name,
        "confidence_level": profile.confidence_level,
        "z_score": profile.z_score,
        "relative_half_width_target": profile.relative_half_width_target,
        "grouping": grouping_name,
        "algorithm": str(key_values.get("algorithm", "ALL")),
        "graph_source": str(key_values.get("graph_source", "ALL")),
        "model": str(key_values.get("model", "ALL")),
        "decoding_condition": str(key_values.get("decoding_condition", "ALL")),
        "metric": str(key_values.get("metric", "ALL")),
        "pair_name": str(key_values.get("pair_name", "ALL")),
        "condition_bits": str(key_values.get("condition_bits", "ALL")),
        "source_finished_run_count": source_finished_run_count,
        "metric_observation_count": int(group["observed_runs"].sum()),
        "condition_count": condition_count,
        "conditions_needing_more_runs": needs_more,
        "condition_share_needing_more_runs": (
            needs_more / condition_count if condition_count else 0.0
        ),
        "conditions_below_expected_replications": below_expected,
        "observed_runs_min": int(group["observed_runs"].min()),
        "observed_runs_median": float(group["observed_runs"].median()),
        "observed_runs_max": int(group["observed_runs"].max()),
        "required_total_runs_max": int(group["required_total_runs"].max()),
        "additional_runs_needed_total": int(group["additional_runs_needed"].sum()),
        "additional_runs_needed_max": int(group["additional_runs_needed"].max()),
    }


def _compact_budget_table(
    *,
    budget_rows: pd.DataFrame,
    observations: pd.DataFrame,
    models: tuple[str, ...],
    profiles: tuple[ReplicationBudgetProfile, ...],
    include_graph_source: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ordered_profiles = tuple(sorted(profiles, key=lambda profile: profile.confidence_level))
    grouping_columns = ["algorithm", "decoding_condition"]
    if include_graph_source:
        grouping_columns = ["algorithm", "graph_source", "decoding_condition"]
    algorithm_decoding_pairs = (
        observations[grouping_columns]
        .drop_duplicates()
        .sort_values(
            by=grouping_columns,
            key=_compact_sort_key,
            kind="mergesort",
        )
    )
    for _, pair in algorithm_decoding_pairs.iterrows():
        algorithm = str(pair["algorithm"])
        graph_source = str(pair["graph_source"]) if include_graph_source else None
        decoding = str(pair["decoding_condition"])
        row: dict[str, object] = {
            "algorithm": _ALGORITHM_LABELS.get(algorithm, algorithm),
        }
        if include_graph_source:
            row["graph_source"] = graph_source
        row["decoding"] = decoding
        for model in models:
            model_prefix = _MODEL_COLUMN_PREFIXES[model]
            model_observations = observations[
                (observations["algorithm"] == algorithm)
                & (observations["decoding_condition"] == decoding)
                & (observations["model"] == model)
            ]
            if include_graph_source:
                model_observations = model_observations[
                    model_observations["graph_source"] == graph_source
                ]
            row[f"{model_prefix}_runs"] = int(model_observations["run_key"].nunique())
            model_budget = budget_rows[
                (budget_rows["algorithm"] == algorithm)
                & (budget_rows["decoding_condition"] == decoding)
                & (budget_rows["model"] == model)
            ]
            if include_graph_source:
                model_budget = model_budget[model_budget["graph_source"] == graph_source]
            row[f"{model_prefix}_condition_metrics"] = int(
                len(model_budget[model_budget["profile"] == profiles[0].name])
            )
            for profile in ordered_profiles:
                profile_budget = model_budget[model_budget["profile"] == profile.name]
                profile_prefix = _profile_column_prefix(profile)
                condition_count = int(len(profile_budget))
                needs_more = int(profile_budget["needs_more_runs"].sum())
                row[f"{model_prefix}_{profile_prefix}_needing_more"] = needs_more
                row[f"{model_prefix}_{profile_prefix}_share"] = (
                    needs_more / condition_count if condition_count else 0.0
                )
                row[f"{model_prefix}_{profile_prefix}_max_required_runs"] = (
                    int(profile_budget["required_total_runs"].max()) if condition_count else 0
                )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _compact_sort_key(values: pd.Series) -> pd.Series:
    if values.name == "algorithm":
        return values.map(lambda value: _ALGORITHM_ORDER.get(str(value), 999))
    if values.name == "decoding_condition":
        return values.map(lambda value: _DECODING_ORDER.get(str(value), 999))
    return values


def _profile_column_prefix(profile: ReplicationBudgetProfile) -> str:
    return f"ci{int(round(profile.confidence_level * 100))}"
