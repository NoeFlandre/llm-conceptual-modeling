"""Data model, decode helpers, and LaTeX rendering for variance decomposition.

All variance-decomposition computation logic (SS decomposition, orthogonal basis
construction) remains in the parent module; this module contains only the stable
data-model surface and pure formatting helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import pandas as pd

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

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


ALGORITHM_SPECS: dict[str, AlgorithmSpec] = {
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


# ---------------------------------------------------------------------------
# Pure decode helpers
# ---------------------------------------------------------------------------


def decode_condition_bits(algorithm: str, condition_bits: object) -> dict[str, int]:
    """Decode a condition-bit string into named factor values.

    Named after the algorithm's ``condition_bit_names``; absent bits default to
    ``-1``.  Returns a dict mapping factor names to ``{1, -1}`` values.
    """
    spec = ALGORITHM_SPECS[algorithm]
    bits = str(condition_bits)
    values: dict[str, int] = {}
    for name, raw_bit in zip(spec.condition_bit_names, bits, strict=False):
        values[name] = 1 if raw_bit == "1" else -1
    for name in spec.condition_bit_names[len(bits):]:
        values[name] = -1
    return values


def decode_decoding_columns(condition_label: str) -> dict[str, object]:
    """Map a decoding-condition label to orthogonal contrast columns."""
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


def coerce_analysis_frame(frame: pd.DataFrame, algorithm: str) -> pd.DataFrame:
    """Ensure ``frame`` contains all decoded factor columns for *algorithm*.

    If the frame already has the factor columns (checked via the first element of
    ``factor_order``) it is returned unchanged.  Otherwise the condition-bit and
    condition-label columns are decoded and the result is returned.
    """
    spec = ALGORITHM_SPECS[algorithm]
    if all(column in frame.columns for column in spec.factor_order):
        return frame.copy()
    if "condition_bits" not in frame.columns or "condition_label" not in frame.columns:
        return frame.copy()

    working = frame.copy()
    decoded_rows = []
    for record in working.to_dict(orient="records"):
        decoded = dict(record)
        decoded.update(decode_condition_bits(algorithm, record.get("condition_bits")))
        decoded.update(decode_decoding_columns(str(record.get("condition_label", ""))))
        decoded_rows.append(decoded)
    return pd.DataFrame(decoded_rows)


# ---------------------------------------------------------------------------
# LaTeX table rendering
# ---------------------------------------------------------------------------


def render_variance_decomposition_table(
    algorithm: str,
    decomposition: pd.DataFrame,
) -> str:
    """Render a variance-decomposition DataFrame as a LaTeX ``table`` environment."""
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
                row_parts.append(
                    _render_metric_cell(
                        decomposition,
                        algorithm,
                        model,
                        metric,
                        feature,
                    )
                )
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
