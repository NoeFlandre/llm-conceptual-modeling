from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_summary_bundle_outputs(
    *,
    output_dir_path: Path,
    manifest_records: list[dict[str, object]],
    overview_records: list[dict[str, object]],
) -> None:
    pd.DataFrame.from_records(manifest_records).to_csv(
        output_dir_path / "bundle_manifest.csv",
        index=False,
    )
    pd.DataFrame.from_records(overview_records).to_csv(
        output_dir_path / "bundle_overview.csv",
        index=False,
    )
    _write_bundle_readme(output_dir_path)


def _write_bundle_readme(output_dir: Path) -> None:
    readme = """# Statistical Reporting Audit Bundle

This directory contains the organized artifacts for the statistical-reporting revision item.

## Purpose

The reviewer asked for confidence intervals and stronger statistical reporting. This bundle captures
the descriptive evidence used to answer that request across all imported evaluated result files.

## Layout

- `bundle_manifest.csv`
  Index of every generated summary file and overview file.
- `bundle_overview.csv`
  One row per algorithm, factor, and metric with global means and per-file winner counts.
- `<algorithm>/<factor_slug>/grouped_metric_summary.csv`
  The full grouped descriptive summary produced by `lcm analyze summary`.
- `<algorithm>/<factor_slug>/metric_overview.csv`
  A compact reviewer-facing overview for that factor.

## Interpretation

The grouped summary files preserve per-source-file provenance. The metric overview files compress
that evidence into global means and file-level winner counts so the revision document can cite the
most informative patterns without repeating entire CSVs inline.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
