from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.common.types import PathLike


def write_variability_bundle(
    *,
    results_root: PathLike,
    output_dir: PathLike,
) -> None:
    """Read pre-computed flat variability CSV files and reorganize into a bundle.

    This mirrors the bundle pattern used in sections 1–4: cross-algorithm summaries
    live at the top level, per-algorithm detail lives in subdirectories, and a
    bundle_manifest.csv + bundle_overview.csv + README.md provide the audit layer.
    """
    results_root_path = Path(results_root)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    manifest_records: list[dict[str, object]] = []
    overview_records: list[dict[str, object]] = []

    # --- Cross-algorithm and model-level summaries (top-level, not algo-specific) ---
    for fname, description in [
        (
            "algorithm_output_variability_summary.csv",
            "Cross-algorithm variability summary: mean Jaccard, exact-match rate, "
            "edge counts, and breadth expansion ratio per algorithm.",
        ),
        (
            "model_output_variability_summary.csv",
            "Model-level variability summary grouped by algorithm and model.",
        ),
        (
            "output_variability_extremes.csv",
            "Most extreme source files by Jaccard and breadth expansion "
            "ratio, per algorithm.",
        ),
    ]:
        src = results_root_path / fname
        if src.exists():
            dest = output_dir_path / fname
            pd.read_csv(src).to_csv(dest, index=False)
            manifest_records.append(
                {
                    "relative_path": fname,
                    "description": description,
                }
            )
            # Build overview from algorithm summary
            if fname == "algorithm_output_variability_summary.csv":
                for _, row in pd.read_csv(src).iterrows():
                    overview_records.append(
                        {
                            "algorithm": row["algorithm"],
                            "mean_pairwise_jaccard": float(row["mean_pairwise_jaccard"]),
                            "exact_match_pair_rate": float(row["exact_match_pair_rate"]),
                            "mean_edge_count": float(row["mean_edge_count"]),
                            "sample_std_edge_count": float(row.get("sample_std_edge_count", 0.0)),
                            "breadth_expansion_ratio": float(row["breadth_expansion_ratio"]),
                        }
                    )

    # --- Per-algorithm subdirectories ---
    for algo in ("algo1", "algo2", "algo3"):
        src_condition = results_root_path / f"{algo}_condition_output_variability.csv"
        if src_condition.exists():
            algo_dir = output_dir_path / algo
            algo_dir.mkdir(parents=True, exist_ok=True)
            pd.read_csv(src_condition).to_csv(
                algo_dir / "condition_output_variability.csv",
                index=False,
            )
            manifest_records.append(
                {
                    "relative_path": f"{algo}/condition_output_variability.csv",
                    "description": (
                        f"Per-model, per-source-file variability metrics for {algo.upper()}. "
                        "Columns: model, source_input, mean_pairwise_jaccard, "
                        "exact_match_pair_rate, mean_edge_count, sample_std_edge_count, "
                        "breadth_expansion_ratio."
                    ),
                }
            )

        # ALGO3-specific breakdowns
        if algo == "algo3":
            src_depth = results_root_path / f"{algo}_output_variability_by_depth.csv"
            if src_depth.exists():
                pd.read_csv(src_depth).to_csv(
                    algo_dir / "by_depth.csv",
                    index=False,
                )
                manifest_records.append(
                    {
                        "relative_path": f"{algo}/by_depth.csv",
                        "description": (
                            "ALGO3 variability broken down by search Depth. "
                            "Columns: Depth, mean_pairwise_jaccard, "
                            "exact_match_pair_rate, mean_edge_count, breadth_expansion_ratio."
                        ),
                    }
                )

            src_words = results_root_path / f"{algo}_output_variability_by_word_count.csv"
            if src_words.exists():
                pd.read_csv(src_words).to_csv(
                    algo_dir / "by_word_count.csv",
                    index=False,
                )
                manifest_records.append(
                    {
                        "relative_path": f"{algo}/by_word_count.csv",
                        "description": (
                            "ALGO3 variability broken down by word budget. "
                            "Columns: Number of Words, mean_pairwise_jaccard, mean_edge_count."
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


def _write_bundle_readme(output_dir: Path) -> None:
    readme = """# Raw Output Variability Audit Bundle

This directory contains the organized artifacts for the output-variability revision item.

## Purpose

The reviewer asked for a mechanism-level explanation of run-to-run variability. This bundle
captures a proxy analysis: do repeated runs produce the same edge sets, or do they drift
into different edges, and if so, how much?

## Layout

- `bundle_manifest.csv`
  Index of every generated file with descriptions.
- `bundle_overview.csv`
  Cross-algorithm summary with mean Jaccard, exact-match rate, edge counts,
  and breadth expansion ratio per algorithm.
- `algorithm_output_variability_summary.csv`
  Top-level cross-algorithm summary.
- `model_output_variability_summary.csv`
  Model-level summary grouped by algorithm and model.
- `output_variability_extremes.csv`
  Most extreme source files by Jaccard and breadth expansion ratio.
- `<algo>/condition_output_variability.csv`
  Per-model, per-source-file variability detail.
- `<algo>/by_depth.csv` (ALGO3 only)
  ALGO3 variability broken down by search Depth.
- `<algo>/by_word_count.csv` (ALGO3 only)
  ALGO3 variability broken down by word budget.

## Key Interpretation

- **ALGO1 and ALGO2**: mean pairwise Jaccard ≈ 1.000, exact-match rate ≈ 0.999,
  breadth expansion ratio ≈ 1.00. These methods are near-deterministic.
- **ALGO3**: mean pairwise Jaccard ≈ 0.077, exact-match rate ≈ 0.001,
  breadth expansion ratio ≈ 4.13. Repeated runs produce substantially different
  edge sets — only about 8% overlap on average.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
