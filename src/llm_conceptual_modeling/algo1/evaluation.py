from llm_conceptual_modeling.common.evaluation_core import evaluate_connection_results_file
from llm_conceptual_modeling.common.types import PathLike


def evaluate_results_file(input_csv_path: PathLike, output_csv_path: PathLike) -> None:
    evaluate_connection_results_file(input_csv_path, output_csv_path)
