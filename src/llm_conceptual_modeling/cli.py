import argparse
from collections.abc import Sequence

from llm_conceptual_modeling.paths import (
    default_results_root,
    default_revision_tracker_root,
)

BASELINE_STRATEGY_CHOICES = [
    "direct-cross-graph",
    "random-uniform-subset",
    "wordnet-ontology-match",
    "edit-distance",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lcm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_subparsers = analyze_parser.add_subparsers(dest="analysis_target", required=True)
    run_parser = subparsers.add_parser("run")
    run_subparsers = run_parser.add_subparsers(dest="run_target", required=True)
    baseline_parser = subparsers.add_parser("baseline")
    baseline_subparsers = baseline_parser.add_subparsers(dest="algorithm", required=True)
    eval_parser = subparsers.add_parser("eval")
    eval_subparsers = eval_parser.add_subparsers(dest="algorithm", required=True)

    factorial_parser = subparsers.add_parser("factorial")
    factorial_subparsers = factorial_parser.add_subparsers(dest="algorithm", required=True)
    doctor_parser = subparsers.add_parser("doctor")
    verify_parser = subparsers.add_parser("verify")
    verify_subparsers = verify_parser.add_subparsers(dest="verify_target", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_subparsers = generate_parser.add_subparsers(dest="algorithm", required=True)

    algo1_parser = eval_subparsers.add_parser("algo1")
    algo1_parser.add_argument("--input", required=True)
    algo1_parser.add_argument("--output", required=True)

    algo2_parser = eval_subparsers.add_parser("algo2")
    algo2_parser.add_argument("--input", required=True)
    algo2_parser.add_argument("--output", required=True)

    algo3_parser = eval_subparsers.add_parser("algo3")
    algo3_parser.add_argument("--input", required=True)
    algo3_parser.add_argument("--output", required=True)

    baseline_algo1_parser = baseline_subparsers.add_parser("algo1")
    baseline_algo1_parser.add_argument("--pair", required=True)
    baseline_algo1_parser.add_argument("--output", required=True)
    baseline_algo1_parser.add_argument(
        "--strategy",
        default="random-uniform-subset",
        choices=BASELINE_STRATEGY_CHOICES,
    )

    baseline_algo2_parser = baseline_subparsers.add_parser("algo2")
    baseline_algo2_parser.add_argument("--pair", required=True)
    baseline_algo2_parser.add_argument("--output", required=True)
    baseline_algo2_parser.add_argument(
        "--strategy",
        default="random-uniform-subset",
        choices=BASELINE_STRATEGY_CHOICES,
    )

    baseline_algo3_parser = baseline_subparsers.add_parser("algo3")
    baseline_algo3_parser.add_argument("--pair", required=True)
    baseline_algo3_parser.add_argument("--output", required=True)
    baseline_algo3_parser.add_argument(
        "--strategy",
        default="random-uniform-subset",
        choices=BASELINE_STRATEGY_CHOICES,
    )

    factorial_algo1_parser = factorial_subparsers.add_parser("algo1")
    factorial_algo1_parser.add_argument("--input", action="append", required=True)
    factorial_algo1_parser.add_argument("--output", required=True)

    factorial_algo2_parser = factorial_subparsers.add_parser("algo2")
    factorial_algo2_parser.add_argument("--input", action="append", required=True)
    factorial_algo2_parser.add_argument("--output", required=True)

    factorial_algo3_parser = factorial_subparsers.add_parser("algo3")
    factorial_algo3_parser.add_argument("--input", required=True)
    factorial_algo3_parser.add_argument("--output", required=True)

    doctor_parser.add_argument("--json", action="store_true")
    doctor_parser.add_argument("--results-root")
    doctor_parser.add_argument("--smoke-root")

    verify_all_parser = verify_subparsers.add_parser("all")
    verify_all_parser.add_argument("--json", action="store_true")

    legacy_parity_parser = verify_subparsers.add_parser("legacy-parity")
    legacy_parity_parser.add_argument("--json", action="store_true")

    for algorithm in ("algo1", "algo2", "algo3"):
        generate_algorithm_parser = generate_subparsers.add_parser(algorithm)
        generate_algorithm_parser.add_argument("--fixture-only", action="store_true")
        generate_algorithm_parser.add_argument("--provider", default="mistral")
        generate_algorithm_parser.add_argument("--config")
        generate_algorithm_parser.add_argument("--json", action="store_true")

    summary_parser = analyze_subparsers.add_parser("summary")
    summary_parser.add_argument("--input", action="append", required=True)
    summary_parser.add_argument("--group-by", action="append", required=True)
    summary_parser.add_argument("--metric", action="append", required=True)
    summary_parser.add_argument("--output", required=True)

    summary_bundle_parser = analyze_subparsers.add_parser("summary-bundle")
    summary_bundle_parser.add_argument("--results-root", default=default_results_root())
    summary_bundle_parser.add_argument("--output-dir", required=True)

    failures_parser = analyze_subparsers.add_parser("failures")
    failures_parser.add_argument("--input", action="append", required=True)
    failures_parser.add_argument("--result-column", required=True)
    failures_parser.add_argument("--output", required=True)

    stability_parser = analyze_subparsers.add_parser("stability")
    stability_parser.add_argument("--input", action="append", required=True)
    stability_parser.add_argument("--group-by", action="append", required=True)
    stability_parser.add_argument("--metric", action="append", required=True)
    stability_parser.add_argument("--output", required=True)

    replication_budget_parser = analyze_subparsers.add_parser("replication-budget")
    replication_budget_parser.add_argument("--input", action="append", required=True)
    replication_budget_parser.add_argument("--output", required=True)
    replication_budget_parser.add_argument(
        "--ci-profile",
        choices=["custom", "strict", "relaxed"],
        default="custom",
    )
    replication_budget_parser.add_argument("--relative-half-width-target", type=float, default=0.05)
    replication_budget_parser.add_argument("--z-score", type=float, default=1.96)

    stability_bundle_parser = analyze_subparsers.add_parser("stability-bundle")
    stability_bundle_parser.add_argument(
        "--results-root",
        default=f"{default_revision_tracker_root()}/replication_stability",
    )
    stability_bundle_parser.add_argument("--output-dir", required=True)

    hypothesis_parser = analyze_subparsers.add_parser("hypothesis")
    hypothesis_parser.add_argument("--input", action="append", required=True)
    hypothesis_parser.add_argument("--factor", required=True)
    hypothesis_parser.add_argument("--pair-by", action="append", required=True)
    hypothesis_parser.add_argument("--metric", action="append", required=True)
    hypothesis_parser.add_argument("--output", required=True)

    hypothesis_bundle_parser = analyze_subparsers.add_parser("hypothesis-bundle")
    hypothesis_bundle_parser.add_argument("--results-root", default=default_results_root())
    hypothesis_bundle_parser.add_argument("--output-dir", required=True)

    output_validity_bundle_parser = analyze_subparsers.add_parser("output-validity-bundle")
    output_validity_bundle_parser.add_argument("--results-root", default=default_results_root())
    output_validity_bundle_parser.add_argument("--output-dir", required=True)

    figures_parser = analyze_subparsers.add_parser("figures")
    figures_parser.add_argument("--input", action="append", required=True)
    figures_parser.add_argument("--id-column", action="append", required=True)
    figures_parser.add_argument("--metric", action="append", required=True)
    figures_parser.add_argument("--output", required=True)

    figures_bundle_parser = analyze_subparsers.add_parser("figures-bundle")
    figures_bundle_parser.add_argument(
        "--results-root",
        default=default_results_root(),
    )
    figures_bundle_parser.add_argument("--output-dir", required=True)

    baseline_comparison_parser = analyze_subparsers.add_parser("baseline-comparison")
    baseline_comparison_parser.add_argument("--input", action="append", required=True)
    baseline_comparison_parser.add_argument("--baseline-input", action="append", required=True)
    baseline_comparison_parser.add_argument("--metric", action="append", required=True)
    baseline_comparison_parser.add_argument("--output", required=True)

    variability_parser = analyze_subparsers.add_parser("variability")
    variability_parser.add_argument("--input", action="append", required=True)
    variability_parser.add_argument("--group-by", action="append", required=True)
    variability_parser.add_argument("--result-column", required=True)
    variability_parser.add_argument("--output", required=True)

    variability_bundle_parser = analyze_subparsers.add_parser("variability-bundle")
    variability_bundle_parser.add_argument(
        "--results-root",
        default=f"{default_revision_tracker_root()}/output_variability",
    )
    variability_bundle_parser.add_argument("--output-dir", required=True)

    baseline_bundle_parser = analyze_subparsers.add_parser("baseline-bundle")
    baseline_bundle_parser.add_argument(
        "--results-root",
        default=default_results_root(),
    )
    baseline_bundle_parser.add_argument("--output-dir", required=True)

    plots_parser = analyze_subparsers.add_parser("plots")
    plots_parser.add_argument("--results-root", required=True)
    plots_parser.add_argument("--output-dir", required=True)

    validate_config_parser = run_subparsers.add_parser("validate-config")
    validate_config_parser.add_argument("--config", required=True)
    validate_config_parser.add_argument("--output-dir", required=True)

    resume_preflight_parser = run_subparsers.add_parser("resume-preflight")
    resume_preflight_parser.add_argument("--config", required=True)
    resume_preflight_parser.add_argument("--repo-root", required=True)
    resume_preflight_parser.add_argument("--results-root")
    resume_preflight_parser.add_argument("--allow-empty", action="store_true")
    resume_preflight_parser.add_argument("--json", action="store_true")

    resume_sweep_parser = run_subparsers.add_parser("resume-sweep")
    resume_sweep_parser.add_argument("--repo-root", required=True)
    resume_sweep_parser.add_argument("--results-root", required=True)
    resume_sweep_parser.add_argument("--json", action="store_true")

    prefetch_runtime_parser = run_subparsers.add_parser("prefetch-runtime")
    prefetch_runtime_parser.add_argument("--config", required=True)
    prefetch_runtime_parser.add_argument("--json", action="store_true")

    status_parser = run_subparsers.add_parser("status")
    status_parser.add_argument("--results-root", required=True)
    status_parser.add_argument("--json", action="store_true")

    drain_remaining_parser = run_subparsers.add_parser("drain-remaining")
    drain_remaining_parser.add_argument("--repo-root", required=True)
    drain_remaining_parser.add_argument("--results-root", required=True)
    drain_remaining_parser.add_argument("--ssh-command", required=True)
    drain_remaining_parser.add_argument("--state-file", required=True)
    drain_remaining_parser.add_argument(
        "--phase",
        choices=("safe", "risky", "all"),
        default="all",
    )
    drain_remaining_parser.add_argument("--full-coverage", action="store_true")
    drain_remaining_parser.add_argument("--root-name-contains")
    drain_remaining_parser.add_argument("--plan-only", action="store_true")
    drain_remaining_parser.add_argument("--poll-seconds", type=int, default=30)
    drain_remaining_parser.add_argument("--stale-after-seconds", type=int, default=3600)
    drain_remaining_parser.add_argument("--quick-resume-script")
    drain_remaining_parser.add_argument("--json", action="store_true")

    drain_status_parser = run_subparsers.add_parser("drain-status")
    drain_status_parser.add_argument("--state-file", required=True)
    drain_status_parser.add_argument("--json", action="store_true")

    smoke_parser = run_subparsers.add_parser("smoke")
    smoke_parser.add_argument("--config", required=True)
    smoke_parser.add_argument("--algorithm", choices=("algo1", "algo2", "algo3"), required=True)
    smoke_parser.add_argument("--model", required=True)
    smoke_parser.add_argument("--pair-name", required=True)
    smoke_parser.add_argument("--condition-bits", required=True)
    smoke_parser.add_argument(
        "--decoding",
        choices=("greedy", "beam", "contrastive"),
        required=True,
    )
    smoke_parser.add_argument("--num-beams", type=int)
    smoke_parser.add_argument("--penalty-alpha", type=float)
    smoke_parser.add_argument("--top-k", type=int, default=4)
    smoke_parser.add_argument("--replication", type=int, default=0)
    smoke_parser.add_argument("--output-root", required=True)
    smoke_parser.add_argument("--resume", action="store_true")
    smoke_parser.add_argument("--dry-run", action="store_true")

    paper_batch_parser = run_subparsers.add_parser("paper-batch")
    paper_batch_parser.add_argument("--provider", default="hf-transformers")
    paper_batch_parser.add_argument("--model", action="append")
    paper_batch_parser.add_argument("--embedding-model")
    paper_batch_parser.add_argument("--output-root")
    paper_batch_parser.add_argument("--config")
    paper_batch_parser.add_argument("--replications", type=int, default=5)
    paper_batch_parser.add_argument("--resume", action="store_true")
    paper_batch_parser.add_argument("--dry-run", action="store_true")

    for algorithm in ("algo1", "algo2", "algo3"):
        algorithm_run_parser = run_subparsers.add_parser(algorithm)
        algorithm_run_parser.add_argument("--provider", default="hf-transformers")
        algorithm_run_parser.add_argument("--model", action="append")
        algorithm_run_parser.add_argument("--embedding-model")
        algorithm_run_parser.add_argument("--output-root")
        algorithm_run_parser.add_argument("--config")
        algorithm_run_parser.add_argument("--replications", type=int, default=5)
        algorithm_run_parser.add_argument("--resume", action="store_true")
        algorithm_run_parser.add_argument("--dry-run", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        from llm_conceptual_modeling.commands.doctor import handle_doctor

        return handle_doctor(args)
    if args.command == "analyze":
        from llm_conceptual_modeling.commands.analyze import handle_analyze

        return handle_analyze(args)
    if args.command == "eval":
        from llm_conceptual_modeling.commands.eval import handle_eval

        return handle_eval(args)
    if args.command == "baseline":
        from llm_conceptual_modeling.commands.baseline import handle_baseline

        return handle_baseline(args)
    if args.command == "factorial":
        from llm_conceptual_modeling.commands.factorial import handle_factorial

        return handle_factorial(args)
    if args.command == "verify":
        from llm_conceptual_modeling.commands.verify import handle_verify

        return handle_verify(args)
    if args.command == "generate":
        from llm_conceptual_modeling.commands.generate import handle_generate

        return handle_generate(args)
    if args.command == "run":
        from llm_conceptual_modeling.commands.run import handle_run

        return handle_run(args)

    parser.error("unsupported command")
    return 2


def run() -> None:
    raise SystemExit(main())
