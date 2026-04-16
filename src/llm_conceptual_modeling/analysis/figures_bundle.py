from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import scipy.stats as stats

from llm_conceptual_modeling.analysis._bundle_stats import _extract_model
from llm_conceptual_modeling.common.types import PathLike


@dataclass(frozen=True)
class FiguresAlgorithmSpec:
    algorithm: str
    metrics: tuple[str, ...]
    id_columns: tuple[str, ...]


_FIGURES_BUNDLE_SPECS: tuple[FiguresAlgorithmSpec, ...] = (
    FiguresAlgorithmSpec(
        algorithm="algo1",
        metrics=("accuracy", "precision", "recall"),
        id_columns=(
            "Repetition",
            "Explanation",
            "Example",
            "Counterexample",
            "Array/List(1/-1)",
            "Tag/Adjacency(1/-1)",
        ),
    ),
    FiguresAlgorithmSpec(
        algorithm="algo2",
        metrics=("accuracy", "precision", "recall"),
        id_columns=(
            "Repetition",
            "Explanation",
            "Example",
            "Counterexample",
            "Array/List(1/-1)",
            "Tag/Adjacency(1/-1)",
            "Convergence",
        ),
    ),
    FiguresAlgorithmSpec(
        algorithm="algo3",
        metrics=("Recall",),
        id_columns=(
            "Repetition",
            "Depth",
            "Example",
            "Counter-Example",
            "Number of Words",
        ),
    ),
)


def write_figures_bundle(
    *,
    results_root: PathLike,
    output_dir: PathLike,
) -> None:
    results_root_path = Path(results_root)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    manifest_records: list[dict[str, object]] = []
    overview_records: list[dict[str, object]] = []

    for algorithm_spec in _FIGURES_BUNDLE_SPECS:
        input_paths = sorted(
            (results_root_path / algorithm_spec.algorithm).glob("*/evaluated/*.csv")
        )
        if not input_paths:
            continue

        # --- Collect per-model frames (keyed by model so all files aggregate) ---
        model_frames: dict[str, list[pd.DataFrame]] = {}

        for input_path in input_paths:
            model = _extract_model(input_path)
            if model not in model_frames:
                model_frames[model] = []

            dataframe = pd.read_csv(input_path)
            df_long = _melt_to_long(
                dataframe,
                model=model,
                source_input=str(input_path),
                algorithm=algorithm_spec.algorithm,
                id_columns=list(algorithm_spec.id_columns),
                metrics=list(algorithm_spec.metrics),
            )
            model_frames[model].append(df_long)

        # --- Per-model: aggregate all files, write distributional summary once ---
        model_all_rows: list[pd.DataFrame] = []
        for model, frames in model_frames.items():
            model_dir = output_dir_path / algorithm_spec.algorithm / model
            model_dir.mkdir(parents=True, exist_ok=True)

            # Aggregate all source files for this model before computing summary
            aggregated = pd.concat(frames, ignore_index=True)
            model_all_rows.append(aggregated)

            dist_summary = _compute_distributional_summary(
                aggregated,
                algorithm=algorithm_spec.algorithm,
                model=model,
            )
            dist_path = model_dir / "distributional_summary.csv"
            dist_summary.to_csv(dist_path, index=False)
            manifest_records.append(
                {
                    "algorithm": algorithm_spec.algorithm,
                    "model": model,
                    "relative_path": (
                        f"{algorithm_spec.algorithm}/{model}/distributional_summary.csv"
                    ),
                    "description": (
                        f"Distributional summary (n, mean, std, 95% CI, median, "
                        f"quartiles, min, max) per metric for {model}, "
                        "aggregated across all source files."
                    ),
                }
            )
            overview_records.extend(dist_summary.to_dict(orient="records"))

        # --- Concatenate all model rows into one long-format file per algorithm ---
        if model_all_rows:
            all_rows = pd.concat(model_all_rows, ignore_index=True)
            metric_rows_path = output_dir_path / f"{algorithm_spec.algorithm}_metric_rows.csv"
            all_rows.to_csv(metric_rows_path, index=False)
            manifest_records.append(
                {
                    "algorithm": algorithm_spec.algorithm,
                    "model": "all",
                    "relative_path": f"{algorithm_spec.algorithm}_metric_rows.csv",
                    "description": (
                        f"Long-format metric rows for all {algorithm_spec.algorithm} models, "
                        "suitable for external plotting tools. Columns: source_input, "
                        "algorithm, model, factor columns, metric, value."
                    ),
                }
            )

    # --- Write bundle metadata ---
    pd.DataFrame.from_records(manifest_records).to_csv(
        output_dir_path / "bundle_manifest.csv",
        index=False,
    )
    if overview_records:
        pd.DataFrame.from_records(overview_records).to_csv(
            output_dir_path / "bundle_overview.csv",
            index=False,
        )
    _write_bundle_readme(output_dir_path)


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
    for (metric), group in df_long.groupby("metric", dropna=False):
        values = group["value"].dropna()
        n = len(values)
        if n == 0:
            continue

        mean_val = float(values.mean())
        std_val = float(values.std(ddof=1)) if n > 1 else 0.0

        # 95% CI using t-distribution (appropriate for small n)
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
def _write_bundle_readme(output_dir: Path) -> None:
    readme = """# Figure-Ready Exports Audit Bundle

This directory contains plot-ready metric exports and distributional summaries for
the figure-ready revision item.

## Purpose

The reviewer asked for confidence intervals and distributional plots alongside mean
performance. This bundle produces deterministic, long-format metric rows and
per-model distributional summaries that can be consumed directly by plotting tools
without ad hoc post-processing.

## Layout

- `bundle_manifest.csv`
  Index of every generated file with descriptions.
- `bundle_overview.csv`
  Distributional summary (n, mean, std, 95% CI, median, q1, q3, min, max) per
  model and metric across all algorithms.
- `<algorithm>_metric_rows.csv`
  Long-format metric rows for all models of that algorithm, suitable for
  box plots, violin plots, and faceted model comparisons.
- `<algorithm>/<model>/distributional_summary.csv`
  Distributional summary for a specific model, including 95% confidence
  intervals on the mean using the t-distribution.

## Interpretation

The distributional summaries show both central tendency (mean, median) and
spread (std, CI, quartiles) per model per metric. When the 95% CI does not
cross zero for a metric, the evaluated performance is reliably above or
below the reference regardless of repetition noise.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
