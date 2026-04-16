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
    coerce_analysis_frame,
    decode_condition_bits,
    decode_decoding_columns,
    render_variance_decomposition_table,
)


def extract_variance_rows_by_algorithm_and_model(
    ledger: dict,
) -> dict[tuple[str, str], list[dict[str, object]]]:
    rows_by_key: dict[tuple[str, str], list[dict[str, object]]] = {}

    for record in ledger.get("records", []):
        if record.get("status") != "finished":
            continue
        identity = record.get("identity", {})
        algorithm = str(identity.get("algorithm", ""))
        spec = ALGORITHM_SPECS.get(algorithm)
        if spec is None:
            continue
        model_full = str(identity.get("model", ""))
        model_label = MODEL_LABELS.get(model_full)
        if model_label is None:
            continue
        condition_label = str(identity.get("condition_label", ""))
        if condition_label not in DECODING_CONDITIONS:
            continue

        metrics = dict(record.get("winner", {}).get("metrics", {}))
        row: dict[str, object] = {
            "algorithm": algorithm,
            "model": model_label,
            "pair_name": str(identity.get("pair_name", "")),
            "condition_bits": str(identity.get("condition_bits", "")),
            "condition_label": condition_label,
            "replication": int(identity.get("replication", 0)),
        }
        row.update(decode_condition_bits(algorithm, row["condition_bits"]))
        row.update(decode_decoding_columns(condition_label))
        for metric in spec.metrics:
            row[metric] = float(metrics.get(metric, 0.0))

        rows_by_key.setdefault((algorithm, model_label), []).append(row)

    return rows_by_key


def compute_variance_decomposition(
    frame: pd.DataFrame,
    algorithm: str,
    model: str,
) -> pd.DataFrame:
    spec = ALGORITHM_SPECS[algorithm]
    working = coerce_analysis_frame(frame, algorithm)
    required_columns = [*spec.factor_order, *spec.metrics]
    _assert_required_columns(working, required_columns)
    _assert_balanced_cells(working, spec.factor_order)

    term_columns = _build_term_columns(working, spec.factor_order)
    _assert_orthogonal_columns(term_columns)

    rows: list[dict[str, object]] = []
    for metric in spec.metrics:
        centered = working[metric].astype(float).to_numpy(dtype=float)
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
    ledger = json.loads((results_root / "ledger.json").read_text(encoding="utf-8"))
    rows_by_key = extract_variance_rows_by_algorithm_and_model(ledger)

    decompositions: list[pd.DataFrame] = []
    for (algorithm, model), rows in sorted(rows_by_key.items()):
        frame = pd.DataFrame(rows)
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


def variance_decomposition_summary(
    results_root: Path,
) -> dict[tuple[str, str], dict[str, dict[str, float]]]:
    bundle = generate_variance_decomposition_bundle(results_root, results_root)
    decomposition = bundle["decomposition"]
    summary: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for (algorithm, model), group in decomposition.groupby(["algorithm", "model"]):
        summary[(algorithm, model)] = {}
        for feature, feature_group in group.groupby("feature"):
            summary[(algorithm, model)][feature] = {
                metric: float(metric_group["pct_with_error"].iloc[0])
                for metric, metric_group in feature_group.groupby("metric")
            }
    return summary
