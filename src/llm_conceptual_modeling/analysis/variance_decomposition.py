"""Deterministic variance decomposition tables for Qwen and Mistral."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from llm_conceptual_modeling.analysis._variance_decomposition_math import (
    _assert_balanced_cells,
    _assert_orthogonal_columns,
    _assert_required_columns,
    _build_term_columns,
    _column_sum_of_squares,
)
from llm_conceptual_modeling.analysis._variance_decomposition_outputs import (
    write_variance_decomposition_outputs,
)
from llm_conceptual_modeling.analysis._variance_decomposition_spec import (
    ALGORITHM_SPECS,
    DECODING_CONDITIONS,
    DEFAULT_OUTPUT_DIRNAME,
    MODEL_LABELS,
    MODEL_ORDER,
    coerce_analysis_frame,
    decode_condition_bits,
    decode_decoding_columns,
    render_variance_decomposition_table,
)

_MODEL_PREFIXES = {
    "Qwen/Qwen3.5-9B": "qwen",
    "Qwen": "qwen",
    "mistralai/Ministral-3-8B-Instruct-2512": "mistral",
    "Mistral": "mistral",
}


def extract_variance_rows_by_algorithm_and_model(
    ledger: dict,
) -> dict[tuple[str, str], list[dict[str, object]]]:
    return _extract_variance_rows_by_algorithm_and_model_from_records(
        ledger.get("records", [])
    )


def build_open_weight_map_extension_summary(frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "algorithm",
        "condition_label",
        "graph_source",
        "pair_name",
        "Example",
        "Number of Words",
        "Depth",
        "model",
        "recall",
    }
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required open-weight summary columns: {missing}")

    working = frame.copy()
    working["model_prefix"] = working["model"].map(_model_prefix)
    working["example"] = working["Example"].astype(int).eq(1)
    working["number_of_words"] = working["Number of Words"].astype(int)
    working["depth"] = working["Depth"].astype(int)
    if working["model_prefix"].isna().any():
        unknown_models = sorted(
            str(model)
            for model in working.loc[working["model_prefix"].isna(), "model"].unique()
        )
        raise ValueError(
            "Unsupported model(s) in open-weight summary frame: "
            f"{', '.join(unknown_models)}"
        )

    grouped = (
        working.groupby(
            [
                "algorithm",
                "condition_label",
                "graph_source",
                "pair_name",
                "example",
                "number_of_words",
                "depth",
                "model_prefix",
            ],
            dropna=False,
        )
        .agg(runs=("recall", "size"), recall=("recall", "mean"))
        .reset_index()
        .sort_values(
            by=[
                "algorithm",
                "condition_label",
                "graph_source",
                "pair_name",
                "example",
                "number_of_words",
                "depth",
                "model_prefix",
            ],
            kind="stable",
        )
    )

    rows: list[dict[str, object]] = []
    summary_keys = [
        "algorithm",
        "condition_label",
        "graph_source",
        "pair_name",
        "example",
        "number_of_words",
        "depth",
    ]
    for key_values, group in grouped.groupby(summary_keys, sort=True, dropna=False):
        algorithm, condition_label, graph_source, pair_name, example, number_of_words, depth = (
            key_values
        )
        row: dict[str, object] = {
            "algorithm": algorithm,
            "condition_label": condition_label,
            "graph_source": graph_source,
            "pair_name": pair_name,
            "example": bool(example),
            "number_of_words": int(number_of_words),
            "depth": int(depth),
        }
        for model_prefix in ("qwen", "mistral"):
            model_group = group[group["model_prefix"] == model_prefix]
            if model_group.empty:
                row[f"{model_prefix}_runs"] = 0
                row[f"{model_prefix}_recall"] = 0.0
                continue
            row[f"{model_prefix}_runs"] = int(model_group["runs"].iloc[0])
            row[f"{model_prefix}_recall"] = float(model_group["recall"].iloc[0])
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(
        by=[
            "algorithm",
            "condition_label",
            "graph_source",
            "pair_name",
            "example",
            "number_of_words",
            "depth",
        ],
        kind="stable",
    ).reset_index(drop=True)


def build_map_recall_summary(frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "graph_source",
        "pair_name",
        "Example",
        "Number of Words",
        "Depth",
        "model",
        "recall",
    }
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required map recall summary columns: {missing}")

    working = frame.copy()
    working["model_prefix"] = working["model"].map(_model_prefix)
    if working["model_prefix"].isna().any():
        unknown_models = sorted(
            str(model)
            for model in working.loc[working["model_prefix"].isna(), "model"].unique()
        )
        raise ValueError(
            "Unsupported model(s) in map recall summary frame: "
            f"{', '.join(unknown_models)}"
        )

    rows: list[dict[str, object]] = []
    for graph_source, group in working.groupby("graph_source", sort=True, dropna=False):
        row: dict[str, object] = {
            "graph_source": graph_source,
            "pair_count": int(group["pair_name"].nunique()),
            "prompt_cell_count": int(
                group[["pair_name", "Example", "Number of Words", "Depth"]]
                .drop_duplicates()
                .shape[0]
            ),
            "overall_runs": int(len(group)),
            "overall_mean_recall": float(group["recall"].mean()),
        }
        for model_prefix in ("qwen", "mistral"):
            model_group = group[group["model_prefix"] == model_prefix]
            row[f"{model_prefix}_runs"] = int(len(model_group))
            row[f"{model_prefix}_mean_recall"] = (
                0.0 if model_group.empty else float(model_group["recall"].mean())
            )
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("graph_source", kind="stable").reset_index(drop=True)
    return summary[
        [
            "graph_source",
            "pair_count",
            "prompt_cell_count",
            "qwen_runs",
            "qwen_mean_recall",
            "mistral_runs",
            "mistral_mean_recall",
            "overall_runs",
            "overall_mean_recall",
        ]
    ]


def compute_variance_decomposition(
    frame: pd.DataFrame,
    algorithm: str,
    model: str,
) -> pd.DataFrame:
    spec = ALGORITHM_SPECS[algorithm]
    working = coerce_analysis_frame(frame, algorithm)
    return _compute_decomposition_rows(
        working,
        algorithm=algorithm,
        model=model,
        factor_order=spec.factor_order,
        metrics=spec.metrics,
    )


def _compute_decomposition_rows(
    frame: pd.DataFrame,
    *,
    algorithm: str,
    model: str,
    factor_order: tuple[str, ...],
    metrics: tuple[str, ...],
) -> pd.DataFrame:
    required_columns = [*factor_order, *metrics]
    _assert_required_columns(frame, required_columns)
    _assert_balanced_cells(frame, factor_order)

    term_columns = _build_term_columns(frame, factor_order)
    _assert_orthogonal_columns(term_columns)

    rows: list[dict[str, object]] = []
    for metric in metrics:
        centered = frame[metric].astype(float).to_numpy(dtype=float)
        centered = centered - centered.mean()
        total_ss = float(np.square(centered).sum())
        if total_ss == 0.0:
            total_ss = 1.0

        effect_rows: list[tuple[str, float]] = []
        for feature_name, columns in term_columns:
            ss_effect = sum(_column_sum_of_squares(column, centered) for column in columns)
            effect_rows.append((feature_name, ss_effect))

        explained_ss = sum(ss for _feature, ss in effect_rows)
        error_ss = max(0.0, total_ss - explained_ss)
        non_error_total = max(0.0, total_ss - error_ss)

        for feature_name, ss_effect in [*effect_rows, ("Error", error_ss)]:
            pct_with_error = (ss_effect / total_ss) * 100.0
            if feature_name == "Error":
                pct_without_error = 0.0
            elif non_error_total == 0.0:
                pct_without_error = 0.0
            else:
                pct_without_error = (ss_effect / non_error_total) * 100.0
            rows.append(
                {
                    "algorithm": algorithm,
                    "model": model,
                    "feature": feature_name,
                    "metric": metric,
                    "ss": ss_effect,
                    "pct_with_error": pct_with_error,
                    "pct_without_error": pct_without_error,
                }
            )

    decomposition = pd.DataFrame(rows)
    return decomposition.sort_values(
        by=["algorithm", "model", "metric", "feature"],
        kind="stable",
    ).reset_index(drop=True)


def generate_variance_decomposition_bundle(
    results_root: Path,
    output_dir: Path | None = None,
) -> dict[str, object]:
    target_dir = output_dir or (results_root / DEFAULT_OUTPUT_DIRNAME)
    target_dir.mkdir(parents=True, exist_ok=True)
    source_frame = _load_variance_source_frame(results_root)

    if "graph_source" in source_frame.columns:
        summary = build_open_weight_map_extension_summary(source_frame)
        summary_csv = target_dir / "open_weight_map_extension_summary.csv"
        summary.to_csv(summary_csv, index=False)
        map_recall_summary = build_map_recall_summary(source_frame)
        map_recall_summary_csv = target_dir / "map_recall_summary.csv"
        map_recall_summary.to_csv(map_recall_summary_csv, index=False)
        output_records = _write_map_extension_variance_decomposition(source_frame, target_dir)
        return {
            "summary_csv": summary_csv,
            "map_recall_summary_csv": map_recall_summary_csv,
            **output_records,
        }

    decompositions: list[pd.DataFrame] = []
    for (algorithm, model), frame in sorted(
        source_frame.groupby(["algorithm", "model"], dropna=False),
        key=lambda item: item[0],
    ):
        frame = frame.copy()
        decompositions.append(compute_variance_decomposition(frame, algorithm, model))
    full = pd.concat(decompositions, ignore_index=True) if decompositions else pd.DataFrame()
    tables: dict[str, str] = {}
    algorithm_csvs: dict[str, Path] = {}
    for algorithm in ALGORITHM_SPECS:
        algorithm_csv_path = target_dir / f"variance_decomposition_{algorithm}.csv"
        algorithm_csvs[algorithm] = algorithm_csv_path
        algorithm_frame = full[full["algorithm"] == algorithm].copy()
        tables[algorithm] = render_variance_decomposition_table(
            algorithm,
            algorithm_frame,
        )

    output_records = write_variance_decomposition_outputs(
        output_dir=target_dir,
        decomposition=full,
        algorithm_csvs=algorithm_csvs,
        tables=tables,
    )

    return {
        "decomposition": full,
        **output_records,
    }


def _load_variance_source_frame(results_root: Path) -> pd.DataFrame:
    map_extension_frame = _load_map_extension_evaluated_frame(results_root)
    if map_extension_frame is not None:
        return map_extension_frame

    batch_summary_path = results_root / "batch_summary.csv"
    if batch_summary_path.exists():
        batch_summary = pd.read_csv(batch_summary_path)
        if not batch_summary.empty and "graph_source" in batch_summary.columns:
            if "raw_row_path" in batch_summary.columns:
                records: list[dict[str, object]] = []
                for record in batch_summary.to_dict(orient="records"):
                    raw_row_path = Path(str(record["raw_row_path"]))
                    resolved_raw_row_path = _resolve_materialized_artifact_path(
                        results_root=results_root,
                        artifact_path=raw_row_path,
                    )
                    if resolved_raw_row_path.exists():
                        raw_row = json.loads(resolved_raw_row_path.read_text(encoding="utf-8"))
                    else:
                        raw_row = {}
                    combined = dict(raw_row)
                    combined.update(record)
                    records.append(combined)
                return pd.DataFrame.from_records(records)
            return batch_summary

    ledger_path = results_root / "ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    return _ledger_to_variance_source_frame(ledger)


def _load_map_extension_evaluated_frame(results_root: Path) -> pd.DataFrame | None:
    evaluated_paths = sorted(results_root.glob("aggregated/algo3/*/combined/evaluated.csv"))
    if not evaluated_paths:
        return None

    evaluated_frames = [
        _normalize_map_extension_frame(pd.read_csv(path), algorithm="algo3")
        for path in evaluated_paths
    ]
    combined = pd.concat(evaluated_frames, ignore_index=True)
    if "graph_source" not in combined.columns:
        return None
    batch_summary_frame = _load_map_extension_batch_summary_frame(results_root)
    if batch_summary_frame is not None:
        _assert_map_extension_recall_matches_batch_summary(combined, batch_summary_frame)
    return combined


def _normalize_map_extension_frame(frame: pd.DataFrame, *, algorithm: str) -> pd.DataFrame:
    normalized = frame.copy()
    rename_map = {
        "Counter-Example": "Counterexample",
        "Recall": "recall",
        "Repetition": "replication",
        "decoding_condition": "condition_label",
    }
    available_rename_map = {
        source_name: target_name
        for source_name, target_name in rename_map.items()
        if source_name in normalized.columns
    }
    normalized = normalized.rename(columns=available_rename_map)
    normalized["algorithm"] = algorithm
    return normalized


def _load_map_extension_batch_summary_frame(results_root: Path) -> pd.DataFrame | None:
    batch_summary_path = results_root / "batch_summary.csv"
    if not batch_summary_path.exists():
        return None
    batch_summary = pd.read_csv(batch_summary_path)
    if batch_summary.empty:
        return None
    normalized = batch_summary.copy()
    bit_series = normalized["condition_bits"].astype(str)
    bit_lengths = sorted(bit_series.str.len().unique().tolist())
    bit_width = max(bit_lengths)
    if bit_width not in {3, 4}:
        raise ValueError(
            f"Unsupported map-extension condition bit width(s): {sorted(bit_lengths)}."
        )
    bits = bit_series.str.zfill(bit_width)
    normalized["Example"] = bits.str[0].map({"0": -1, "1": 1}).astype(int)
    if bit_width == 4:
        normalized["Counterexample"] = bits.str[1].map({"0": -1, "1": 1}).astype(int)
        normalized["Number of Words"] = bits.str[2].map({"0": 3, "1": 5}).astype(int)
        normalized["Depth"] = bits.str[3].map({"0": 1, "1": 2}).astype(int)
    else:
        normalized["Counterexample"] = -1
        normalized["Number of Words"] = bits.str[1].map({"0": 3, "1": 5}).astype(int)
        normalized["Depth"] = bits.str[2].map({"0": 1, "1": 2}).astype(int)
    return _normalize_map_extension_frame(normalized, algorithm="algo3")


def _assert_map_extension_recall_matches_batch_summary(
    evaluated_frame: pd.DataFrame,
    batch_summary_frame: pd.DataFrame,
) -> None:
    identity_columns = [
        "algorithm",
        "model",
        "graph_source",
        "pair_name",
        "condition_label",
        "Example",
        "Counterexample",
        "Number of Words",
        "Depth",
        "replication",
    ]
    _assert_required_columns(evaluated_frame, [*identity_columns, "recall"])
    _assert_required_columns(batch_summary_frame, [*identity_columns, "recall"])

    evaluated = evaluated_frame[identity_columns + ["recall"]].copy()
    batch_summary = batch_summary_frame[identity_columns + ["recall"]].copy()
    evaluated = evaluated.rename(columns={"recall": "evaluated_recall"})
    batch_summary = batch_summary.rename(columns={"recall": "batch_summary_recall"})

    merged = evaluated.merge(
        batch_summary,
        on=identity_columns,
        how="outer",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        mismatch_count = int((merged["_merge"] != "both").sum())
        raise ValueError(
            "Map-extension evaluated/batch-summary identity mismatch for "
            f"{mismatch_count} run(s)."
        )
    mismatched = merged[
        ~np.isclose(
            merged["evaluated_recall"].astype(float),
            merged["batch_summary_recall"].astype(float),
        )
    ]
    if not mismatched.empty:
        raise ValueError(
            "Map-extension recall mismatch between evaluated artifacts and batch summary "
            f"for {len(mismatched)} run(s)."
        )


def _ledger_to_variance_source_frame(ledger: dict[str, object]) -> pd.DataFrame:
    rows_by_key = _extract_variance_rows_by_algorithm_and_model_from_records(
        ledger.get("records", [])
    )
    rows: list[dict[str, object]] = []
    for key_rows in rows_by_key.values():
        rows.extend(key_rows)
    return pd.DataFrame.from_records(rows)


def _resolve_materialized_artifact_path(*, results_root: Path, artifact_path: Path) -> Path:
    if artifact_path.exists():
        return artifact_path
    if not artifact_path.is_absolute():
        return results_root / artifact_path
    try:
        relative_path = artifact_path.relative_to(Path("/workspace/results"))
    except ValueError:
        return artifact_path
    candidate_paths = [results_root / relative_path]
    if relative_path.parts and relative_path.parts[0] == results_root.name:
        candidate_paths.append(results_root / Path(*relative_path.parts[1:]))
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path
    return artifact_path


def _write_map_extension_variance_decomposition(
    source_frame: pd.DataFrame,
    target_dir: Path,
) -> dict[str, object]:
    normalized = _normalize_map_extension_frame(source_frame, algorithm="algo3")
    required_columns = [
        "graph_source",
        "pair_name",
        "model",
        "Example",
        "Number of Words",
        "Depth",
        "recall",
    ]
    _assert_required_columns(normalized, required_columns)

    decompositions: list[pd.DataFrame] = []
    for model_name, model_frame in sorted(
        normalized.groupby("model", dropna=False),
        key=lambda item: str(item[0]),
    ):
        model_label = MODEL_LABELS.get(str(model_name), str(model_name))
        working = model_frame.copy()
        working["Number of Words"] = _binary_contrast_levels(working["Number of Words"])
        working["Depth"] = _binary_contrast_levels(working["Depth"])
        decomposition = _compute_decomposition_rows(
            working,
            algorithm="algo3",
            model=model_label,
            factor_order=("graph_source", "pair_name", "Example", "Number of Words", "Depth"),
            metrics=("recall",),
        )
        decompositions.append(decomposition)

    full = pd.concat(decompositions, ignore_index=True) if decompositions else pd.DataFrame()
    algorithm_csvs = {"algo3": target_dir / "variance_decomposition_algo3.csv"}
    tables = {"algo3": render_variance_decomposition_table("algo3", full)}
    output_records = write_variance_decomposition_outputs(
        output_dir=target_dir,
        decomposition=full,
        algorithm_csvs=algorithm_csvs,
        tables=tables,
    )
    output_records["map_decomposition_csvs"] = _write_map_specific_decompositions(
        normalized,
        target_dir,
    )
    return output_records


def _write_map_specific_decompositions(
    normalized: pd.DataFrame,
    target_dir: Path,
) -> dict[str, Path]:
    map_decomposition_csvs: dict[str, Path] = {}
    for graph_source, graph_frame in sorted(
        normalized.groupby("graph_source", dropna=False),
        key=lambda item: str(item[0]),
    ):
        decompositions: list[pd.DataFrame] = []
        for model_name, model_frame in sorted(
            graph_frame.groupby("model", dropna=False),
            key=lambda item: str(item[0]),
        ):
            model_label = MODEL_LABELS.get(str(model_name), str(model_name))
            working = model_frame.copy()
            working["Number of Words"] = _binary_contrast_levels(working["Number of Words"])
            working["Depth"] = _binary_contrast_levels(working["Depth"])
            decomposition = _compute_decomposition_rows(
                working,
                algorithm="algo3",
                model=model_label,
                factor_order=("pair_name", "Example", "Number of Words", "Depth"),
                metrics=("recall",),
            )
            decomposition.insert(0, "graph_source", graph_source)
            decompositions.append(decomposition)
        map_decomposition = (
            pd.concat(decompositions, ignore_index=True) if decompositions else pd.DataFrame()
        )
        output_path = target_dir / f"variance_decomposition_{graph_source}.csv"
        map_decomposition.to_csv(output_path, index=False)
        map_decomposition_csvs[str(graph_source)] = output_path
    return map_decomposition_csvs


def _model_prefix(model: object) -> str | float:
    if not isinstance(model, str):
        return float("nan")
    prefix = _MODEL_PREFIXES.get(model)
    if prefix is None:
        return float("nan")
    return prefix


def _binary_contrast_levels(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise")
    unique_values = sorted(pd.unique(numeric))
    if unique_values == [-1, 1]:
        return numeric.astype(int)
    if len(unique_values) != 2:
        raise ValueError(f"Expected exactly two levels for contrast coding, got {unique_values}.")
    low_value, high_value = unique_values
    return numeric.map({low_value: -1, high_value: 1}).astype(int)


def _extract_variance_rows_by_algorithm_and_model_from_records(
    records: list[dict[str, object]] | tuple[dict[str, object], ...] | list[object],
) -> dict[tuple[str, str], list[dict[str, object]]]:
    rows_by_key: dict[tuple[str, str], list[dict[str, object]]] = {}

    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("status", "finished") != "finished":
            continue

        if "identity" in record:
            identity = record.get("identity", {})
            if not isinstance(identity, dict):
                continue
            metrics = record.get("winner", {})
            if not isinstance(metrics, dict):
                metrics = {}
            metric_values = metrics.get("metrics", {})
            if not isinstance(metric_values, dict):
                metric_values = {}
        else:
            identity = record
            metric_values = record

        algorithm = str(identity.get("algorithm", ""))
        spec = ALGORITHM_SPECS.get(algorithm)
        if spec is None:
            continue

        model_full = str(identity.get("model", ""))
        model_label = MODEL_LABELS.get(model_full)
        if model_label is None and model_full in MODEL_ORDER:
            model_label = model_full
        if model_label is None:
            continue

        condition_label = str(identity.get("condition_label", ""))
        if condition_label not in DECODING_CONDITIONS:
            continue

        condition_bits = str(identity.get("condition_bits", ""))
        row: dict[str, object] = {
            "algorithm": algorithm,
            "model": model_label,
            "pair_name": str(identity.get("pair_name", "")),
            "condition_bits": condition_bits,
            "condition_label": condition_label,
            "replication": int(identity.get("replication", 0)),
        }
        if "graph_source" in identity or "graph_source" in record:
            row["graph_source"] = str(identity.get("graph_source", record.get("graph_source", "")))
        row.update(decode_condition_bits(algorithm, condition_bits))
        row.update(decode_decoding_columns(condition_label))
        for metric in spec.metrics:
            value = metric_values.get(metric, record.get(metric, 0.0))
            row[metric] = float(value)

        rows_by_key.setdefault((algorithm, model_label), []).append(row)

    return rows_by_key
