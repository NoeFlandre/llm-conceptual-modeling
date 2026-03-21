from collections.abc import Sequence

from llm_conceptual_modeling.common.factorial_core import run_multi_metric_factorial_analysis
from llm_conceptual_modeling.common.types import MultiMetricFactorialSpec, PathLike

SPEC = MultiMetricFactorialSpec(
    factor_columns=[
        "Explanation",
        "Example",
        "Counterexample",
        "Array/List(1/-1)",
        "Tag/Adjacency(1/-1)",
    ],
    metric_columns=["accuracy", "recall", "precision"],
    output_columns=["accuracy", "recall", "precision", "Feature"],
)


def run_factorial_analysis(input_csv_paths: Sequence[PathLike], output_path: PathLike) -> None:
    run_multi_metric_factorial_analysis(list(input_csv_paths), output_path, SPEC)
