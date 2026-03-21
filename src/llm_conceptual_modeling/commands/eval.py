from argparse import Namespace

from llm_conceptual_modeling.algo1.evaluation import evaluate_results_file as evaluate_algo1_results
from llm_conceptual_modeling.algo2.evaluation import evaluate_results_file as evaluate_algo2_results
from llm_conceptual_modeling.algo3.evaluation import evaluate_results_file as evaluate_algo3_results


def handle_eval(args: Namespace) -> int:
    handlers = {
        "algo1": evaluate_algo1_results,
        "algo2": evaluate_algo2_results,
        "algo3": evaluate_algo3_results,
    }
    handlers[args.algorithm](args.input, args.output)
    return 0
