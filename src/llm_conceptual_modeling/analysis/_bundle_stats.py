"""Pure DataFrame transformers for output-validity and breadth bundle assembly."""
from __future__ import annotations

from pathlib import PurePosixPath

import pandas as pd


def _annotate(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = df.copy()
    out["summary_type"] = label
    return out


def _extract_model(source_input: str) -> str:
    parts = PurePosixPath(source_input).parts
    for i, part in enumerate(parts):
        if part in ("results", "legacy") and i + 2 < len(parts):
            return str(parts[i + 2])
    for i, part in enumerate(parts):
        if part in ("raw", "evaluated") and i > 0:
            return str(parts[i - 1])
    return str(parts[-1]) if parts else source_input


def _build_validity_summary(df: pd.DataFrame, algorithm: str) -> pd.DataFrame:
    df = df.copy()
    df["model"] = df["source_input"].apply(_extract_model)
    records: list[dict[str, object]] = []
    for model, model_frame in df.groupby("model", dropna=False):
        valid = int((model_frame["failure_category"] == "valid_output").sum())
        empty = int((model_frame["failure_category"] == "empty_output").sum())
        malformed = int((model_frame["failure_category"] == "malformed_output").sum())
        total = int(len(model_frame))
        records.append(
            {
                "algorithm": algorithm,
                "model": str(model),
                "total_rows": total,
                "valid_count": valid,
                "empty_count": empty,
                "malformed_count": malformed,
                "failure_count": empty + malformed,
                "failure_rate": (empty + malformed) / total if total > 0 else 0.0,
            }
        )
    return pd.DataFrame.from_records(records)


def _build_breadth_distribution(df: pd.DataFrame, algorithm: str) -> pd.DataFrame:
    valid = df[df["failure_category"] == "valid_output"].copy()
    valid["model"] = valid["source_input"].apply(_extract_model)
    records: list[dict[str, object]] = []
    for model, model_frame in valid.groupby("model", dropna=False):
        counts = model_frame["parsed_edge_count"]
        mean_val = float(counts.mean())
        median_val = float(counts.median())
        records.append(
            {
                "algorithm": algorithm,
                "model": str(model),
                "valid_row_count": int(len(counts)),
                "mean": mean_val,
                "median": median_val,
                "std": float(counts.std()),
                "min": int(counts.min()),
                "max": int(counts.max()),
                "mean_minus_median": mean_val - median_val,
            }
        )
    return pd.DataFrame.from_records(records)


def _build_parsed_edge_quartiles(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["failure_category"] == "valid_output"].copy()
    records: list[dict[str, object]] = []
    for keys, group in valid.groupby(["algorithm", "model"], dropna=False):
        algorithm_val, model = keys
        counts = group["parsed_edge_count"]
        q1 = float(counts.quantile(0.25))
        q2 = float(counts.quantile(0.50))
        q3 = float(counts.quantile(0.75))
        records.append(
            {
                "algorithm": str(algorithm_val),
                "model": str(model),
                "q1": q1,
                "q2": q2,
                "q3": q3,
                "p90": float(counts.quantile(0.90)),
                "p95": float(counts.quantile(0.95)),
                "p99": float(counts.quantile(0.99)),
                "iqr": q3 - q1,
            }
        )
    return pd.DataFrame.from_records(records)


def _build_failure_rates(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for keys, group in df.groupby(["algorithm", "model"], dropna=False):
        algorithm_val, model = keys  # type: ignore[assignment]
        total = int(len(group))
        failures = int((group["failure_category"] != "valid_output").sum())
        records.append(
            {
                "algorithm": str(algorithm_val),
                "model": str(model),
                "total_rows": total,
                "failure_count": failures,
                "failure_rate": failures / total if total > 0 else 0.0,
            }
        )
    return pd.DataFrame.from_records(records)


def _build_parsed_edge_counts(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["failure_category"] == "valid_output"].copy()
    records: list[dict[str, object]] = []
    for keys, group in valid.groupby(["algorithm", "model"], dropna=False):
        algorithm_val, model = keys
        counts = group["parsed_edge_count"]
        mean_val = float(counts.mean())
        median_val = float(counts.median())
        records.append(
            {
                "algorithm": str(algorithm_val),
                "model": str(model),
                "valid_row_count": int(len(counts)),
                "mean": round(mean_val, 2),
                "median": median_val,
                "min": int(counts.min()),
                "max": int(counts.max()),
            }
        )
    return pd.DataFrame.from_records(records)
