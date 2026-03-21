import sys
from argparse import Namespace

from llm_conceptual_modeling.algo1.baseline import write_baseline_results_file as write_algo1
from llm_conceptual_modeling.algo2.baseline import write_baseline_results_file as write_algo2
from llm_conceptual_modeling.algo3.baseline import write_baseline_results_file as write_algo3


def handle_baseline(args: Namespace) -> int:
    try:
        if args.algorithm == "algo1":
            write_algo1(args.pair, args.output)
            return 0
        if args.algorithm == "algo2":
            write_algo2(args.pair, args.output)
            return 0
        if args.algorithm == "algo3":
            write_algo3(args.pair, args.output)
            return 0
        raise ValueError(f"Unsupported baseline algorithm: {args.algorithm}")
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
