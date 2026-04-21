"""Pure helper utilities extracted from stability_bundle.py for cohesion."""

from __future__ import annotations

import pandas as pd


def _slugify(value: str) -> str:
    """Lowercase slug of a factor label for use in filenames."""
    slug = value.lower()
    for source, target in (
        (" ", "_"),
        ("/", "_"),
        ("(", ""),
        (")", ""),
        ("-", "_"),
    ):
        slug = slug.replace(source, target)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def frame_to_overview_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Convert a metric stability overview DataFrame to overview record dicts."""
    records: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        records.append(
            {
                "algorithm": row["algorithm"],
                "metric": row["metric"],
                "condition_count": int(row["condition_count"]),
                "mean_cv": float(row["mean_cv"]),
                "median_cv": float(row["median_cv"]),
                "max_cv": float(row["max_cv"]),
                "mean_range_width": float(row["mean_range_width"]),
                "max_range_width": float(row["max_range_width"]),
            }
        )
    return records


def patch_algorithm_rows(
    frame: pd.DataFrame,
    replacement_row: dict[str, object],
) -> pd.DataFrame:
    """Replace the row matching (algorithm, metric) in frame with replacement_row."""
    if frame.empty:
        return frame
    if "algorithm" not in frame.columns or "metric" not in frame.columns:
        return frame
    algorithm = replacement_row["algorithm"]
    metric = replacement_row["metric"]
    filtered = frame.loc[~((frame["algorithm"] == algorithm) & (frame["metric"] == metric))].copy()
    replacement_frame = pd.DataFrame.from_records([replacement_row])
    return pd.concat([filtered, replacement_frame], ignore_index=True)
