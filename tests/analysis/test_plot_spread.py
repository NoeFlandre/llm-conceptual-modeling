from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis._plot_spread import write_main_metric_spread_plots


def test_write_main_metric_spread_plots_writes_box_and_violin_files(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "plots"
    output_dir.mkdir()

    frame = pd.DataFrame(
        {
            "algorithm": ["algo1", "algo1", "algo2", "algo2"],
            "model": ["gpt-5", "gpt-5", "gpt-5", "gpt-5"],
            "metric": ["accuracy", "precision", "accuracy", "precision"],
            "value": [0.8, 0.4, 0.7, 0.5],
        }
    )

    write_main_metric_spread_plots(
        frame=frame,
        boxplot_output_path=output_dir / "main_metric_spread_boxplots.png",
        violin_output_path=output_dir / "main_metric_spread_violins.png",
    )

    assert (output_dir / "main_metric_spread_boxplots.png").exists()
    assert (output_dir / "main_metric_spread_violins.png").exists()
