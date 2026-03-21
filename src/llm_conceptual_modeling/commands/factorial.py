from argparse import Namespace

from llm_conceptual_modeling.algo1.factorial import run_factorial_analysis as run_algo1_factorial
from llm_conceptual_modeling.algo2.factorial import run_factorial_analysis as run_algo2_factorial
from llm_conceptual_modeling.algo3.factorial import run_factorial_analysis as run_algo3_factorial


def handle_factorial(args: Namespace) -> int:
    handlers = {
        "algo1": run_algo1_factorial,
        "algo2": run_algo2_factorial,
        "algo3": run_algo3_factorial,
    }
    handlers[args.algorithm](args.input, args.output)
    return 0
