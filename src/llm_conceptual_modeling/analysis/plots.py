from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis._path_helpers import _discover_main_results_root
from llm_conceptual_modeling.analysis._plot_distribution import (
    write_distribution_plot,
)
from llm_conceptual_modeling.analysis._plot_factor_effect import (
    write_factor_effect_plot,
)
from llm_conceptual_modeling.analysis._plot_frames import (
    _build_aggregated_distribution_frame,
    _build_aggregated_factor_effect_frame,
    _build_aggregated_variability_frame,
    _build_main_metric_rows,
)
from llm_conceptual_modeling.analysis._plot_spread import (
    write_main_metric_spread_plots,
)
from llm_conceptual_modeling.analysis._plot_variability import (
    write_variability_plot,
)
from llm_conceptual_modeling.common.types import PathLike


def write_revision_plots(*, results_root: PathLike, output_dir: PathLike) -> None:
    results_root_path = Path(results_root)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    if (results_root_path / "figure_exports" / "bundle_overview.csv").exists():
        figures = pd.read_csv(results_root_path / "figure_exports" / "bundle_overview.csv")
        hypothesis = pd.read_csv(results_root_path / "hypothesis_testing" / "bundle_overview.csv")
        variability = pd.read_csv(results_root_path / "output_variability" / "bundle_overview.csv")
    else:
        figures = _build_aggregated_distribution_frame(results_root_path)
        hypothesis = _build_aggregated_factor_effect_frame(results_root_path)
        variability = _build_aggregated_variability_frame(results_root_path)
    main_metric_rows = _build_main_metric_rows(_discover_main_results_root(results_root_path))

    write_distribution_plot(figures, output_dir_path / "distribution_metrics.png")
    write_factor_effect_plot(hypothesis, output_dir_path / "factor_effect_summary.png")
    write_variability_plot(variability, output_dir_path / "raw_output_variability.png")
    write_main_metric_spread_plots(
        frame=main_metric_rows,
        boxplot_output_path=output_dir_path / "main_metric_spread_boxplots.png",
        violin_output_path=output_dir_path / "main_metric_spread_violins.png",
    )
