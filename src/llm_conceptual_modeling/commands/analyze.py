import sys
from argparse import Namespace

from llm_conceptual_modeling.analysis.baseline_bundle import write_baseline_comparison_bundle
from llm_conceptual_modeling.analysis.baseline_comparison import write_baseline_metric_comparison
from llm_conceptual_modeling.analysis.failures import write_failure_analysis
from llm_conceptual_modeling.analysis.figures import write_figure_ready_metric_rows
from llm_conceptual_modeling.analysis.figures_bundle import write_figures_bundle
from llm_conceptual_modeling.analysis.hypothesis import write_paired_factor_hypothesis_tests
from llm_conceptual_modeling.analysis.hypothesis_bundle import write_hypothesis_testing_bundle
from llm_conceptual_modeling.analysis.output_validity_bundle import write_output_validity_bundle
from llm_conceptual_modeling.analysis.stability import write_grouped_metric_stability
from llm_conceptual_modeling.analysis.stability_bundle import write_stability_bundle
from llm_conceptual_modeling.analysis.summary import write_grouped_metric_summary
from llm_conceptual_modeling.analysis.summary_bundle import write_statistical_reporting_bundle
from llm_conceptual_modeling.analysis.variability import write_output_variability_analysis
from llm_conceptual_modeling.analysis.variability_bundle import write_variability_bundle


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
        if args.analysis_target == "summary-bundle":
            write_statistical_reporting_bundle(
                results_root=args.results_root,
                output_dir=args.output_dir,
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
        if args.analysis_target == "stability-bundle":
            write_stability_bundle(
                results_root=args.results_root,
                output_dir=args.output_dir,
            )
            return 0
        if args.analysis_target == "hypothesis":
            write_paired_factor_hypothesis_tests(
                args.input,
                args.output,
                factor=args.factor,
                pair_by=args.pair_by,
                metrics=args.metric,
            )
            return 0
        if args.analysis_target == "hypothesis-bundle":
            write_hypothesis_testing_bundle(
                results_root=args.results_root,
                output_dir=args.output_dir,
            )
            return 0
        if args.analysis_target == "output-validity-bundle":
            write_output_validity_bundle(
                results_root=args.results_root,
                output_dir=args.output_dir,
            )
            return 0
        if args.analysis_target == "figures":
            write_figure_ready_metric_rows(
                args.input,
                args.output,
                id_columns=args.id_column,
                metrics=args.metric,
            )
            return 0
        if args.analysis_target == "figures-bundle":
            write_figures_bundle(
                results_root=args.results_root,
                output_dir=args.output_dir,
            )
            return 0
        if args.analysis_target == "baseline-comparison":
            write_baseline_metric_comparison(
                args.input,
                args.baseline_input,
                args.output,
                metrics=args.metric,
            )
            return 0
        if args.analysis_target == "variability":
            write_output_variability_analysis(
                args.input,
                args.output,
                group_by=args.group_by,
                result_column=args.result_column,
            )
            return 0
        if args.analysis_target == "variability-bundle":
            write_variability_bundle(
                results_root=args.results_root,
                output_dir=args.output_dir,
            )
            return 0
        if args.analysis_target == "baseline-bundle":
            write_baseline_comparison_bundle(
                results_root=args.results_root,
                output_dir=args.output_dir,
            )
            return 0
        raise ValueError(f"Unsupported analyze target: {args.analysis_target}")
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
