from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis._bundle_stats import (
    _annotate,
    _build_breadth_distribution,
    _build_failure_rates,
    _build_parsed_edge_counts,
    _build_parsed_edge_quartiles,
    _build_validity_summary,
    _extract_model,
)
from llm_conceptual_modeling.analysis._output_validity_outputs import (
    write_output_validity_bundle_outputs,
)
from llm_conceptual_modeling.analysis.failures import write_failure_analysis
from llm_conceptual_modeling.common.types import PathLike


@dataclass(frozen=True)
class OutputValidityAlgorithmSpec:
    algorithm: str
    result_column: str


_OUTPUT_VALIDITY_BUNDLE_SPECS: tuple[OutputValidityAlgorithmSpec, ...] = (
    OutputValidityAlgorithmSpec(algorithm="algo1", result_column="Result"),
    OutputValidityAlgorithmSpec(algorithm="algo2", result_column="Result"),
    OutputValidityAlgorithmSpec(algorithm="algo3", result_column="Results"),
)


def write_output_validity_bundle(
    *,
    results_root: PathLike,
    output_dir: PathLike,
) -> None:
    """Generate the organized output-validity bundle.

    Produces:
        <output_dir>/
            README.md
            bundle_manifest.csv
            bundle_overview.csv
            failure_rates.csv
            parsed_edge_counts.csv
            parsed_edge_quartiles.csv
            algo1/
                row_level_validity.csv
                validity_summary.csv
                breadth_distribution.csv
            algo2/
                row_level_validity.csv
                validity_summary.csv
                breadth_distribution.csv
            algo3/
                row_level_validity.csv
                validity_summary.csv
                breadth_distribution.csv
    """
    results_root_path = Path(results_root)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    manifest_records: list[dict[str, object]] = []
    overview_records: list[dict[str, object]] = []

    all_row_frames: list[pd.DataFrame] = []

    for algo_spec in _OUTPUT_VALIDITY_BUNDLE_SPECS:
        raw_glob_root = results_root_path / algo_spec.algorithm
        raw_input_paths = sorted(raw_glob_root.glob("*/raw/*.csv"))
        if not raw_input_paths:
            raise ValueError(
                f"No raw CSVs found for {algo_spec.algorithm} under {results_root}"
            )

        algo_output_dir = output_dir_path / algo_spec.algorithm
        algo_output_dir.mkdir(parents=True, exist_ok=True)

        # Row-level classification
        row_level_output = algo_output_dir / "row_level_validity.csv"
        write_failure_analysis(
            [str(p) for p in raw_input_paths],
            row_level_output,
            result_column=algo_spec.result_column,
        )

        # Compute validity summary per model for this algorithm
        classified = pd.read_csv(row_level_output)
        classified["algorithm"] = algo_spec.algorithm
        classified["model"] = classified["source_input"].apply(_extract_model)
        all_row_frames.append(classified)

        validity_summary = _build_validity_summary(classified, algo_spec.algorithm)
        validity_summary_path = algo_output_dir / "validity_summary.csv"
        validity_summary.to_csv(validity_summary_path, index=False)

        # Compute breadth distribution per model for this algorithm
        breadth_dist = _build_breadth_distribution(classified, algo_spec.algorithm)
        breadth_dist_path = algo_output_dir / "breadth_distribution.csv"
        breadth_dist.to_csv(breadth_dist_path, index=False)

        manifest_records.append(
            {
                "algorithm": algo_spec.algorithm,
                "file": str(algo_output_dir / "row_level_validity.csv"),
                "description": "One row per raw output, classified as valid/empty/malformed",
            }
        )
        manifest_records.append(
            {
                "algorithm": algo_spec.algorithm,
                "file": str(validity_summary_path),
                "description": "Validity counts per model (valid/empty/malformed rows)",
            }
        )
        manifest_records.append(
            {
                "algorithm": algo_spec.algorithm,
                "file": str(breadth_dist_path),
                "description": "Edge count statistics per model: mean, median, std, min, max",
            }
        )

        overview_records.extend(validity_summary.to_dict(orient="records"))
        overview_records.extend(breadth_dist.to_dict(orient="records"))

    # Global aggregate files
    combined = pd.concat(all_row_frames, ignore_index=True)

    failure_rates = _build_failure_rates(combined)
    failure_rates.to_csv(output_dir_path / "failure_rates.csv", index=False)
    manifest_records.append(
        {
            "algorithm": "all",
            "file": str(output_dir_path / "failure_rates.csv"),
            "description": "Failure rate per algorithm-model combination across all algorithms",
        }
    )

    parsed_edge_counts = _build_parsed_edge_counts(combined)
    parsed_edge_counts.to_csv(output_dir_path / "parsed_edge_counts.csv", index=False)
    manifest_records.append(
        {
            "algorithm": "all",
            "file": str(output_dir_path / "parsed_edge_counts.csv"),
            "description": "Parsed edge count statistics per algorithm-model",
        }
    )

    parsed_edge_quartiles = _build_parsed_edge_quartiles(combined)
    parsed_edge_quartiles.to_csv(output_dir_path / "parsed_edge_quartiles.csv", index=False)
    manifest_records.append(
        {
            "algorithm": "all",
            "file": str(output_dir_path / "parsed_edge_quartiles.csv"),
            "description": "Edge count quartile and percentile distribution per algorithm-model",
        }
    )

    bundle_overview = pd.concat(
        [
            _annotate(failure_rates, "failure_rates"),
            _annotate(parsed_edge_counts, "parsed_edge_counts"),
            _annotate(parsed_edge_quartiles, "parsed_edge_quartiles"),
        ],
        ignore_index=True,
    )
    write_output_validity_bundle_outputs(
        output_dir_path=output_dir_path,
        manifest_records=manifest_records,
        overview_records=bundle_overview,
        failure_rates=failure_rates,
        parsed_edge_counts=parsed_edge_counts,
        parsed_edge_quartiles=parsed_edge_quartiles,
    )
