from llm_conceptual_modeling.algo1.evaluation import (
    evaluate_results_file as evaluate_algo1_results_file,
)
from llm_conceptual_modeling.common.types import PathLike


def evaluate_results_file(input_csv_path: PathLike, output_csv_path: PathLike) -> None:
    evaluate_algo1_results_file(input_csv_path, output_csv_path)
