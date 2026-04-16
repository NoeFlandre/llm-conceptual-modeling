from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_conceptual_modeling.analysis._plot_factor_effect import (
    write_factor_effect_plot,
)


def test_write_factor_effect_plot_writes_png(tmp_path: Path) -> None:
    output_path = tmp_path / "factor_effect.png"
    frame = pd.DataFrame(
        {
            "algorithm": ["algo1", "algo2"],
            "factor": ["Explanation", "Example"],
            "metric": ["precision", "accuracy"],
            "mean_difference_average": [0.02, 0.03],
            "significant_share": [0.5, 0.83],
        }
    )

    write_factor_effect_plot(frame=frame, output_path=output_path)

    assert output_path.exists()
