from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis._plot_variability import (
    write_variability_plot,
)


def test_write_variability_plot_writes_png(tmp_path: Path) -> None:
    output_path = tmp_path / "variability.png"
    frame = pd.DataFrame(
        {
            "algorithm": ["algo1", "algo3"],
            "model": ["gpt-5", "gpt-5"],
            "mean_pairwise_jaccard": [0.998, 0.077],
            "breadth_expansion_ratio": [1.00, 4.13],
        }
    )

    write_variability_plot(frame=frame, output_path=output_path)

    assert output_path.exists()
