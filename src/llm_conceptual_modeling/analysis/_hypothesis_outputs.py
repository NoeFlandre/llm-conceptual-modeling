from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_hypothesis_bundle_outputs(
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
    readme = """# Hypothesis-Testing Audit Bundle

This directory contains the organized artifacts for the formal hypothesis-testing revision item.

## Purpose

The reviewer asked for p-values, multiple-comparison adjustment, and stronger inferential support.
This bundle captures those paired tests in a structure that can be read by factor rather than as a
flat directory of unrelated CSV files.

## Layout

- `bundle_manifest.csv`
  Index of every generated paired-test file, significance-summary file, and factor overview file.
- `bundle_overview.csv`
  One row per algorithm, factor, and metric with significant-test counts, direction counts, and
  the strongest adjusted result.
- `<algorithm>/<factor_slug>/paired_tests.csv`
  Full paired t-test output for that factor.
- `<algorithm>/<factor_slug>/significance_summary.csv`
  Count summary by metric, direction, and adjusted-significance status.
- `<algorithm>/<factor_slug>/factor_overview.csv`
  Compact reviewer-facing overview for that factor.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
