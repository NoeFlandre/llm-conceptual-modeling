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
from llm_conceptual_modeling.analysis.plots import write_revision_plots
from llm_conceptual_modeling.analysis.replication_budget import write_replication_budget_analysis
from llm_conceptual_modeling.analysis.replication_budget_summary import (
    write_compact_replication_budget_sufficiency_table,
    write_replication_budget_sufficiency_summary,
)
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
        if args.analysis_target == "replication-budget":
            if args.ci_profile == "strict":
                args.relative_half_width_target = 0.05
                args.z_score = 1.96
            elif args.ci_profile == "relaxed":
                args.relative_half_width_target = 0.1
                args.z_score = 1.645
            write_replication_budget_analysis(
                args.input,
                args.output,
                relative_half_width_target=args.relative_half_width_target,
                z_score=args.z_score,
            )
            return 0
        if args.analysis_target == "replication-budget-sufficiency":
            write_replication_budget_sufficiency_summary(
                results_root=args.results_root,
                output_csv_path=args.output,
                models=tuple(args.model) if args.model else None,
                expected_replications=args.expected_replications,
            )
            if args.compact_output:
                write_compact_replication_budget_sufficiency_table(
                    results_root=args.results_root,
                    output_csv_path=args.compact_output,
                    models=tuple(args.model) if args.model else None,
                    expected_replications=args.expected_replications,
                    include_graph_source=args.include_graph_source,
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
        if args.analysis_target == "plots":
            write_revision_plots(
                results_root=args.results_root,
                output_dir=args.output_dir,
            )
            return 0
        raise ValueError(f"Unsupported analyze target: {args.analysis_target}")
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
