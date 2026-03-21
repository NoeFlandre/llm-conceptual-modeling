from pathlib import Path

from llm_conceptual_modeling.algo1.evaluation import (
    evaluate_results_file as evaluate_algo1_results_file,
)


def evaluate_results_file(input_csv_path: str | Path, output_csv_path: str | Path) -> None:
    evaluate_algo1_results_file(input_csv_path, output_csv_path)
