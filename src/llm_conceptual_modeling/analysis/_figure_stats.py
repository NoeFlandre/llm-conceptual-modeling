from __future__ import annotations

from typing import Any

import pandas as pd
import scipy.stats as stats


def _melt_to_long(
    df: pd.DataFrame,
    *,
    model: str,
    source_input: str,
    algorithm: str,
    id_columns: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    melted = df[id_columns + list(metrics)].melt(
        id_vars=id_columns,
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )
    melted.insert(0, "model", model)
    melted.insert(0, "algorithm", algorithm)
    melted.insert(0, "source_input", source_input)
    return melted


def _compute_distributional_summary(
    df_long: pd.DataFrame,
    *,
    algorithm: str,
    model: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for metric, group in df_long.groupby("metric", dropna=False):
        values = group["value"].dropna()
        n = len(values)
        if n == 0:
            continue

        mean_val = float(values.mean())
        std_val = float(values.std(ddof=1)) if n > 1 else 0.0

        if n >= 2:
            ci_low, ci_high = _mean_ci95(values)
        else:
            ci_low = ci_high = mean_val

        sorted_vals = values.sort_values().tolist()
        median_val = float(sorted_vals[n // 2]) if n > 0 else float("nan")
        q1_val = float(sorted_vals[max(0, int(n * 0.25) - 1)]) if n >= 4 else float("nan")
        q3_val = float(sorted_vals[min(n - 1, int(n * 0.75) - 1)]) if n >= 4 else float("nan")
        min_val = float(sorted_vals[0]) if sorted_vals else float("nan")
        max_val = float(sorted_vals[-1]) if sorted_vals else float("nan")

        records.append(
            {
                "algorithm": algorithm,
                "model": model,
                "metric": metric,
                "n": n,
                "mean": round(mean_val, 6),
                "sample_std": round(std_val, 6),
                "ci95_low": round(ci_low, 6),
                "ci95_high": round(ci_high, 6),
                "median": round(median_val, 6),
                "q1": round(q1_val, 6) if not pd.isna(q1_val) else None,
                "q3": round(q3_val, 6) if not pd.isna(q3_val) else None,
                "min": round(min_val, 6) if not pd.isna(min_val) else None,
                "max": round(max_val, 6) if not pd.isna(max_val) else None,
            }
        )
    return pd.DataFrame.from_records(records) if records else pd.DataFrame()


def _mean_ci95(series: pd.Series) -> tuple[float, float]:
    n = len(series)
    mean = series.mean()
    se = series.std(ddof=1) / (n**0.5)
    t_val = float(stats.t.ppf(0.975, df=n - 1))
    return float(mean - t_val * se), float(mean + t_val * se)
