from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis._plot_distribution import (
    write_distribution_plot,
)


def test_write_distribution_plot_writes_png(tmp_path: Path) -> None:
    output_path = tmp_path / "distribution.png"
    frame = pd.DataFrame(
        {
            "algorithm": ["algo1", "algo1"],
            "model": ["gpt-5", "gpt-5"],
            "metric": ["accuracy", "precision"],
            "mean": [0.9, 0.4],
            "ci95_low": [0.88, 0.38],
            "ci95_high": [0.92, 0.42],
        }
    )

    write_distribution_plot(frame=frame, output_path=output_path)

    assert output_path.exists()
