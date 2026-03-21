import sys
from argparse import Namespace

from llm_conceptual_modeling.analysis.failures import write_failure_analysis
from llm_conceptual_modeling.analysis.stability import write_grouped_metric_stability
from llm_conceptual_modeling.analysis.summary import write_grouped_metric_summary


def handle_analyze(args: Namespace) -> int:
    try:
        if args.analysis_target == "summary":
            write_grouped_metric_summary(
                args.input,
                args.output,
                group_by=args.group_by,
                metrics=args.metric,
            )
            return 0
        if args.analysis_target == "failures":
            write_failure_analysis(
                args.input,
                args.output,
                result_column=args.result_column,
            )
            return 0
        if args.analysis_target == "stability":
            write_grouped_metric_stability(
                args.input,
                args.output,
                group_by=args.group_by,
                metrics=args.metric,
            )
            return 0
        raise ValueError(f"Unsupported analyze target: {args.analysis_target}")
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
