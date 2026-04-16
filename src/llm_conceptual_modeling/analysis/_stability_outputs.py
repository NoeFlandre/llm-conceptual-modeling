from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis._algo3_stability import Algo3PairAwareOutputs
from llm_conceptual_modeling.analysis._stability_helpers import (
    frame_to_overview_records,
    patch_algorithm_rows,
)


def write_stability_bundle_outputs(
    *,
    output_dir_path: Path,
    manifest_records: list[dict[str, object]],
    overview_records: list[dict[str, object]],
    replication_budget_overview_records: list[dict[str, object]],
    source_variability_frame: pd.DataFrame | None,
    source_overall_frame: pd.DataFrame | None,
    algo3_pair_aware_outputs: Algo3PairAwareOutputs | None,
) -> None:
    variability_incidence_path = output_dir_path / "variability_incidence_by_algorithm.csv"
    overall_stability_path = output_dir_path / "overall_metric_stability_by_algorithm.csv"

    if source_overall_frame is not None:
        patched_overall_frame = source_overall_frame.copy()
        if algo3_pair_aware_outputs is not None:
            patched_overall_frame = patch_algorithm_rows(
                patched_overall_frame,
                algo3_pair_aware_outputs.overall_row,
            )
        patched_overall_frame.to_csv(overall_stability_path, index=False)
        overview_records.extend(
            frame_to_overview_records(patched_overall_frame)
        )

    if source_variability_frame is not None:
        patched_variability_frame = source_variability_frame.copy()
        if algo3_pair_aware_outputs is not None:
            patched_variability_frame = patch_algorithm_rows(
                patched_variability_frame,
                algo3_pair_aware_outputs.variability_row,
            )
        patched_variability_frame.to_csv(variability_incidence_path, index=False)

    pd.DataFrame.from_records(manifest_records).to_csv(
        output_dir_path / "bundle_manifest.csv",
        index=False,
    )
    if overview_records:
        pd.DataFrame.from_records(overview_records).to_csv(
            output_dir_path / "bundle_overview.csv",
            index=False,
        )
    if replication_budget_overview_records:
        pd.DataFrame.from_records(replication_budget_overview_records).to_csv(
            output_dir_path / "replication_budget_overview.csv",
            index=False,
        )
        manifest_records.append(
            {
                "algorithm": "cross_algorithm",
                "factor": "overall",
                "relative_path": "replication_budget_overview.csv",
                "description": (
                    "Conservative required-run summary by algorithm and metric under the "
                    "95% CI precision calculation. Key columns: max_required_total_runs, "
                    "max_additional_runs_needed, conditions_needing_more_runs."
                ),
            }
        )
        pd.DataFrame.from_records(manifest_records).to_csv(
            output_dir_path / "bundle_manifest.csv",
            index=False,
        )
    _write_bundle_readme(output_dir_path)


def _write_bundle_readme(output_dir: Path) -> None:
    readme = """# Replication Stability Audit Bundle

This directory contains the organized artifacts for the replication-stability revision item.

## Purpose

The reviewer asked for a principled justification of the five-replication decision. This bundle
captures repetition-level stability analysis over the five recorded runs in the imported corpus.
It now also adds a conservative confidence-interval run calculation using

  n = ((1.96 * s) / (r * |x_bar|))^2

with a 95% CI (`1.96`) and a 5% relative half-width target (`r = 0.05`).

## Layout

- `bundle_manifest.csv`
  Index of every generated file with descriptions.
- `bundle_overview.csv`
  Cross-algorithm stability summary (CV and range-width) by algorithm and metric.
- `variability_incidence_by_algorithm.csv`
  Cross-algorithm count and share of conditions that changed across repetitions.
- `overall_metric_stability_by_algorithm.csv`
  Coefficient-of-variation and range-width summaries by algorithm and metric.
- `replication_budget_overview.csv`
  Conservative required-run summary by algorithm and metric under the CI precision rule.
- `<algorithm>/condition_stability.csv`
  Per-file, per-condition stability statistics across the five repetitions.
- `<algorithm>/replication_budget_by_condition.csv`
  Per-condition required total runs and additional runs needed under the CI precision rule.
- `<algorithm>/<factor>_stability_by_level.csv`
  Aggregated stability summaries grouped by a specific factor level.
- `<algorithm>/<factor>_variability_incidence.csv`
  Incidence of any run-to-run variation grouped by a specific factor level.

## Key Interpretation

- **ALGO1 and ALGO2 are nearly repetition-stable**: median CVs are 0.0 across most conditions.
- **ALGO2 is especially stable when Convergence = 1**: zero varying conditions.
- **ALGO3 is orders of magnitude noisier**: median CV on recall is 3.87, meaning noise is
  nearly 4 times the signal size — not a small or marginal difference.
- **The CI-based run budget is conservative**: low-variance conditions can be justified with the
  existing runs, while low-mean high-variance conditions can demand many more.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
