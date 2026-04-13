"""Deterministic variance decomposition tables for Qwen and Mistral."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_LABELS = {
    "Qwen/Qwen3.5-9B": "Qwen",
    "mistralai/Ministral-3-8B-Instruct-2512": "Mistral",
}

DECODING_CONDITIONS = (
    "greedy",
    "beam_num_beams_2",
    "beam_num_beams_6",
    "contrastive_penalty_alpha_0.2",
    "contrastive_penalty_alpha_0.8",
)

MODEL_ORDER = ("Qwen", "Mistral")
DEFAULT_OUTPUT_DIRNAME = "variance_decomposition"

DECODING_FACTOR_NAMES = (
    "Greedy vs Beam/Contrastive",
    "Beam vs Contrastive",
    "Beam Width",
    "Contrastive Penalty",
)


@dataclass(frozen=True)
class AlgorithmSpec:
    metrics: tuple[str, ...]
    condition_bit_names: tuple[str, ...]
    factor_order: tuple[str, ...]


ALGORITHM_SPECS = {
    "algo1": AlgorithmSpec(
        metrics=("accuracy", "recall", "precision"),
        condition_bit_names=(
            "Explanation",
            "Example",
            "Counterexample",
            "Array/List",
            "Tag/Adjacency",
        ),
        factor_order=(
            "Explanation",
            "Example",
            "Counterexample",
            "Array/List",
            "Tag/Adjacency",
            *DECODING_FACTOR_NAMES,
        ),
    ),
    "algo2": AlgorithmSpec(
        metrics=("accuracy", "recall", "precision"),
        condition_bit_names=(
            "Explanation",
            "Example",
            "Counterexample",
            "Array/List",
            "Tag/Adjacency",
            "Convergence",
        ),
        factor_order=(
            "Explanation",
            "Example",
            "Counterexample",
            "Array/List",
            "Tag/Adjacency",
            "Convergence",
            *DECODING_FACTOR_NAMES,
        ),
    ),
    "algo3": AlgorithmSpec(
        metrics=("recall",),
        condition_bit_names=(
            "Example",
            "Counterexample",
            "Number of Words",
            "Depth",
        ),
        factor_order=(
            "Example",
            "Counterexample",
            "Number of Words",
            "Depth",
            *DECODING_FACTOR_NAMES,
        ),
    ),
}


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
        row.update(_decode_condition_bits(algorithm, row["condition_bits"]))
        row.update(_decode_decoding_columns(condition_label))
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
    working = _coerce_analysis_frame(frame, algorithm)
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


def render_variance_decomposition_table(
    algorithm: str,
    decomposition: pd.DataFrame,
) -> str:
    spec = ALGORITHM_SPECS[algorithm]
    metrics = spec.metrics
    features = _feature_order(algorithm)
    column_spec = "l|" + "|".join("l" * len(metrics) for _ in MODEL_ORDER)
    width = "1.15\\textwidth" if algorithm == "algo3" else "1.4\\textwidth"
    caption = _table_caption(algorithm)

    lines = [
        "\\begin{table}[h]",
        "\\begin{adjustwidth}{-2.25in}{0in}",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\resizebox{{{width}}}{{!}}{{%",
        f"\\begin{{tabular}}{{|{column_spec}|}}",
        "\\hline",
    ]

    model_header = " & ".join(
        f"\\multicolumn{{{len(metrics)}}}{{c|}}{{\\textit{{{model}}}}}" for model in MODEL_ORDER
    )
    lines.append(f"\\multicolumn{{1}}{{|l|}}{{\\textbf{{Feature}}}} & {model_header} \\\\ \\hline")

    metric_header_parts = ["\\multicolumn{1}{|c|}{}"]
    for _model in MODEL_ORDER:
        for metric in metrics:
            metric_header_parts.append(
                f"\\multicolumn{{1}}{{c|}}{{\\textbf{{{_latex_escape(metric)}}}}}"
            )
    lines.append(" & ".join(metric_header_parts) + " \\\\ \\hline")

    for index, feature in enumerate(features):
        row_parts = [_latex_feature_label(feature)]
        for model in MODEL_ORDER:
            for metric in metrics:
                row_parts.append(_render_metric_cell(decomposition, algorithm, model, metric, feature))
        lines.append(" & ".join(row_parts) + " \\\\")
        if index < len(features) - 1:
            lines.append("\\hline")

    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "}",
            f"\\label{{table:{algorithm}_variance_decomposition}}",
            "\\end{adjustwidth}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


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

    csv_path = target_dir / "variance_decomposition.csv"
    full.to_csv(csv_path, index=False)

    tables: dict[str, str] = {}
    algorithm_csvs: dict[str, Path] = {}
    for algorithm in ALGORITHM_SPECS:
        algorithm_frame = full[full["algorithm"] == algorithm].copy()
        algorithm_csv_path = target_dir / f"variance_decomposition_{algorithm}.csv"
        algorithm_frame.to_csv(algorithm_csv_path, index=False)
        algorithm_csvs[algorithm] = algorithm_csv_path
        table = render_variance_decomposition_table(
            algorithm,
            algorithm_frame,
        )
        tables[algorithm] = table
        (target_dir / f"variance_decomposition_{algorithm}.tex").write_text(
            table,
            encoding="utf-8",
        )

    combined = "\n\n".join(tables[algorithm] for algorithm in ALGORITHM_SPECS)
    combined_path = target_dir / "variance_decomposition.tex"
    combined_path.write_text(combined, encoding="utf-8")

    return {
        "decomposition": full,
        "decomposition_csv": csv_path,
        "algorithm_csvs": algorithm_csvs,
        "tables": tables,
        "combined_table": combined_path,
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


def _decode_condition_bits(algorithm: str, condition_bits: object) -> dict[str, int]:
    spec = ALGORITHM_SPECS[algorithm]
    bits = str(condition_bits)
    values: dict[str, int] = {}
    for name, raw_bit in zip(spec.condition_bit_names, bits, strict=False):
        values[name] = 1 if raw_bit == "1" else -1
    for name in spec.condition_bit_names[len(bits) :]:
        values[name] = -1
    return values


def _decode_decoding_columns(condition_label: str) -> dict[str, object]:
    if condition_label == "greedy":
        return {
            "Decoding Family": "greedy",
            "Greedy vs Beam/Contrastive": 4,
            "Beam vs Contrastive": 0,
            "Beam Width": 0,
            "Contrastive Penalty": 0,
        }
    if condition_label.startswith("beam_"):
        return {
            "Decoding Family": "beam",
            "Greedy vs Beam/Contrastive": -1,
            "Beam vs Contrastive": 1,
            "Beam Width": 1 if condition_label.endswith("_6") else -1,
            "Contrastive Penalty": 0,
        }
    return {
        "Decoding Family": "contrastive",
        "Greedy vs Beam/Contrastive": -1,
        "Beam vs Contrastive": -1,
        "Beam Width": 0,
        "Contrastive Penalty": 1 if condition_label.endswith("_0.8") else -1,
    }


def _build_term_columns(
    frame: pd.DataFrame,
    factor_order: tuple[str, ...],
) -> list[tuple[str, list[np.ndarray]]]:
    basis = {factor_name: _basis_columns_for_factor(frame, factor_name) for factor_name in factor_order}
    term_columns: list[tuple[str, list[np.ndarray]]] = []
    for factor_name in factor_order:
        term_columns.append((factor_name, basis[factor_name]))
    for left_name, right_name in combinations(factor_order, 2):
        if left_name in DECODING_FACTOR_NAMES and right_name in DECODING_FACTOR_NAMES:
            continue
        interaction_columns: list[np.ndarray] = []
        for left_column in basis[left_name]:
            for right_column in basis[right_name]:
                interaction_columns.append(left_column * right_column)
        term_columns.append((f"{left_name} & {right_name}", interaction_columns))
    return term_columns


def _basis_columns_for_factor(frame: pd.DataFrame, factor_name: str) -> list[np.ndarray]:
    column = frame[factor_name].astype(float).to_numpy(dtype=float)
    return [column]


def _column_sum_of_squares(column: np.ndarray, centered_response: np.ndarray) -> float:
    denominator = float(np.dot(column, column))
    if denominator == 0.0:
        return 0.0
    numerator = float(np.dot(column, centered_response))
    return (numerator**2) / denominator


def _assert_required_columns(frame: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        msg = f"Missing required variance decomposition columns: {missing}"
        raise ValueError(msg)


def _assert_balanced_cells(frame: pd.DataFrame, factor_order: tuple[str, ...]) -> None:
    counts = frame.groupby(list(factor_order), dropna=False).size()
    if counts.empty:
        msg = "Cannot decompose variance from an empty design."
        raise ValueError(msg)
    if counts.nunique() != 1:
        msg = "Variance decomposition requires a balanced design across factor cells."
        raise ValueError(msg)


def _assert_orthogonal_columns(term_columns: list[tuple[str, list[np.ndarray]]]) -> None:
    all_columns: list[np.ndarray] = []
    for _feature_name, columns in term_columns:
        all_columns.extend(columns)
    for left_index, left in enumerate(all_columns):
        for right in all_columns[left_index + 1 :]:
            dot = float(np.dot(left, right))
            if abs(dot) > 1e-8:
                msg = "Variance decomposition basis is not orthogonal."
                raise ValueError(msg)


def _feature_order(algorithm: str) -> list[str]:
    main_factors = list(ALGORITHM_SPECS[algorithm].factor_order)
    interactions = [
        f"{left_name} & {right_name}"
        for left_name, right_name in combinations(main_factors, 2)
        if not (left_name in DECODING_FACTOR_NAMES and right_name in DECODING_FACTOR_NAMES)
    ]
    return [*main_factors, *interactions, "Error"]


def _table_caption(algorithm: str) -> str:
    metric_phrase = "recall" if algorithm == "algo3" else "accuracy, recall, and precision"
    algo_number = algorithm.replace("algo", "")
    return (
        "Deterministic variance decomposition across all modeled factors for "
        f"Qwen and Mistral in Algorithm {algo_number}. "
        f"Each cell reports the share of variance for {metric_phrase}. "
        "When accounting for the error term changes the displayed share, the table shows "
        "\\textit{without error vs. with error}."
    )


def _render_metric_cell(
    decomposition: pd.DataFrame,
    algorithm: str,
    model: str,
    metric: str,
    feature: str,
) -> str:
    match = decomposition[
        (decomposition["algorithm"] == algorithm)
        & (decomposition["model"] == model)
        & (decomposition["metric"] == metric)
        & (decomposition["feature"] == feature)
    ]
    if match.empty:
        return ""
    row = match.iloc[0]
    with_error = float(row["pct_with_error"])
    without_error = float(row["pct_without_error"])
    if feature == "Error":
        return f"{with_error:.2f}"
    if round(without_error, 2) != round(with_error, 2):
        return f"{without_error:.2f} vs {with_error:.2f}"
    return f"{with_error:.2f}"


def _latex_feature_label(feature: str) -> str:
    if feature == "Error":
        return "\\cellcolor[HTML]{FDE9D9}Error term"
    return _latex_escape(feature).replace(" & ", r"\&")


def _latex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def _coerce_analysis_frame(frame: pd.DataFrame, algorithm: str) -> pd.DataFrame:
    spec = ALGORITHM_SPECS[algorithm]
    if all(column in frame.columns for column in spec.factor_order):
        return frame.copy()
    if "condition_bits" not in frame.columns or "condition_label" not in frame.columns:
        return frame.copy()

    working = frame.copy()
    decoded_rows = []
    for record in working.to_dict(orient="records"):
        decoded = dict(record)
        decoded.update(_decode_condition_bits(algorithm, record.get("condition_bits")))
        decoded.update(_decode_decoding_columns(str(record.get("condition_label", ""))))
        decoded_rows.append(decoded)
    return pd.DataFrame(decoded_rows)
