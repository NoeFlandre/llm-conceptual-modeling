from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis._color_mapping import _canonical_model_label
from llm_conceptual_modeling.analysis._path_helpers import _path_triplet


def _build_aggregated_distribution_frame(results_root: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    evaluated_paths = list(results_root.glob("aggregated/*/*/combined/evaluated.csv"))
    if not evaluated_paths:
        evaluated_paths = list(results_root.glob("aggregated/*/*/*/evaluated.csv"))
    for evaluated_path in evaluated_paths:
        algorithm, model, _condition = _path_triplet(evaluated_path)
        frame = pd.read_csv(evaluated_path)
        metric_columns = [
            column for column in ("accuracy", "precision", "recall", "Recall") if column in frame
        ]
        for metric in metric_columns:
            series = frame[metric]
            for value in series.tolist():
                records.append(
                    {
                        "algorithm": algorithm,
                        "model": model,
                        "metric": metric,
                        "value": float(value),
                    }
                )
    return pd.DataFrame.from_records(records)


def _build_aggregated_factor_effect_frame(results_root: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    factorial_paths = list(results_root.glob("aggregated/*/*/combined/factorial.csv"))
    if not factorial_paths:
        factorial_paths = list(results_root.glob("aggregated/*/*/*/factorial.csv"))
    for factorial_path in factorial_paths:
        algorithm, model, _condition = _path_triplet(factorial_path)
        if algorithm not in {"algo1", "algo2"}:
            continue
        frame = pd.read_csv(factorial_path)
        feature_column = "Feature" if "Feature" in frame.columns else "feature"
        metric_columns = [
            column for column in ("accuracy", "precision", "recall") if column in frame.columns
        ]
        filtered = frame[
            ~frame[feature_column].isin(["Error", "Repetition"])
            & ~frame[feature_column].astype(str).str.contains("_AND_", regex=False)
        ]
        for _, row in filtered.iterrows():
            for metric in metric_columns:
                records.append(
                    {
                        "algorithm": algorithm,
                        "model": model,
                        "factor": row[feature_column],
                        "metric": metric,
                        "effect_share": float(row[metric]),
                    }
                )
    if not records:
        return pd.DataFrame(
            columns=[
                "algorithm",
                "factor",
                "metric",
                "effect_share",
            ]
        )
    return pd.DataFrame.from_records(records)


def _build_aggregated_variability_frame(results_root: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    variability_paths = list(results_root.glob("aggregated/*/*/combined/output_variability.csv"))
    if not variability_paths:
        variability_paths = list(results_root.glob("aggregated/*/*/*/output_variability.csv"))
    for variability_path in variability_paths:
        algorithm, model, _condition = _path_triplet(variability_path)
        frame = pd.read_csv(variability_path)
        if frame.empty:
            continue
        records.append(
            {
                "algorithm": algorithm,
                "model": model,
                "mean_pairwise_jaccard": float(frame["mean_pairwise_jaccard"].mean()),
                "breadth_expansion_ratio": float(
                    frame["union_edge_count"].mean()
                    / frame["mean_edge_count"].clip(lower=1.0).mean()
                )
                if "union_edge_count" in frame and "mean_edge_count" in frame
                else float(frame["breadth_expansion_ratio"].mean()),
            }
        )
    return pd.DataFrame.from_records(records)


def _build_main_metric_rows(results_root: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    metric_map = {
        "algo1": ("accuracy", "precision", "recall"),
        "algo2": ("accuracy", "precision", "recall"),
        "algo3": ("Recall",),
    }
    for algorithm, metrics in metric_map.items():
        for evaluated_path in sorted((results_root / algorithm).glob("*/evaluated/*.csv")):
            model = _canonical_model_label(evaluated_path.parts[-3])
            frame = pd.read_csv(evaluated_path)
            for metric in metrics:
                if metric not in frame.columns:
                    continue
                normalized_metric = metric.lower()
                for value in frame[metric].dropna().tolist():
                    records.append(
                        {
                            "algorithm": algorithm,
                            "model": model,
                            "metric": normalized_metric,
                            "value": float(value),
                        }
                    )
    return pd.DataFrame.from_records(records)
