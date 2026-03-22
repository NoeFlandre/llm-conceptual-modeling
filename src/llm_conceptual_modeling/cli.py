import argparse
from collections.abc import Sequence

from llm_conceptual_modeling.commands.analyze import handle_analyze
from llm_conceptual_modeling.commands.audit import handle_audit
from llm_conceptual_modeling.commands.baseline import handle_baseline
from llm_conceptual_modeling.commands.doctor import handle_doctor
from llm_conceptual_modeling.commands.eval import handle_eval
from llm_conceptual_modeling.commands.factorial import handle_factorial
from llm_conceptual_modeling.commands.generate import handle_generate
from llm_conceptual_modeling.commands.probe import handle_probe
from llm_conceptual_modeling.commands.verify import handle_verify


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lcm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_subparsers = analyze_parser.add_subparsers(dest="analysis_target", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_subparsers = audit_parser.add_subparsers(dest="audit_target", required=True)
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
    probe_parser = subparsers.add_parser("probe")
    probe_subparsers = probe_parser.add_subparsers(dest="algorithm", required=True)

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

    baseline_algo2_parser = baseline_subparsers.add_parser("algo2")
    baseline_algo2_parser.add_argument("--pair", required=True)
    baseline_algo2_parser.add_argument("--output", required=True)

    baseline_algo3_parser = baseline_subparsers.add_parser("algo3")
    baseline_algo3_parser.add_argument("--pair", required=True)
    baseline_algo3_parser.add_argument("--output", required=True)

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

    verify_all_parser = verify_subparsers.add_parser("all")
    verify_all_parser.add_argument("--json", action="store_true")

    legacy_parity_parser = verify_subparsers.add_parser("legacy-parity")
    legacy_parity_parser.add_argument("--json", action="store_true")

    for algorithm in ("algo1", "algo2", "algo3"):
        generate_algorithm_parser = generate_subparsers.add_parser(algorithm)
        generate_algorithm_parser.add_argument("--fixture-only", action="store_true")
        generate_algorithm_parser.add_argument("--json", action="store_true")
        generate_algorithm_parser.add_argument("--model")
        generate_algorithm_parser.add_argument("--pair")
        generate_algorithm_parser.add_argument("--output-root")
        generate_algorithm_parser.add_argument(
            "--provider", default="mistral", choices=["mistral", "anthropic"],
            help="DEBUG: 'anthropic' is for research/debugging only, not part of paper pipeline"
        )
        generate_algorithm_parser.add_argument("--replications", type=int, default=5)
        generate_algorithm_parser.add_argument("--resume", action="store_true")
    generate_subparsers.choices["algo2"].add_argument("--embedding-model")

    probe_algo1_parser = probe_subparsers.add_parser("algo1")
    probe_algo1_parser.add_argument("--run-name", required=True)
    probe_algo1_parser.add_argument("--model", required=True)
    probe_algo1_parser.add_argument(
        "--provider", default="mistral", choices=["mistral", "anthropic"],
        help="DEBUG: 'anthropic' is for research/debugging only, not part of paper pipeline"
    )
    probe_algo1_parser.add_argument("--subgraph1-edge", action="append", required=True)
    probe_algo1_parser.add_argument("--subgraph2-edge", action="append", required=True)
    probe_algo1_parser.add_argument("--output-dir", required=True)
    probe_algo1_parser.add_argument("--resume", action="store_true")

    probe_algo2_parser = probe_subparsers.add_parser("algo2")
    probe_algo2_parser.add_argument("--run-name", required=True)
    probe_algo2_parser.add_argument("--model", required=True)
    probe_algo2_parser.add_argument(
        "--provider", default="mistral", choices=["mistral", "anthropic"],
        help="DEBUG: 'anthropic' is for research/debugging only, not part of paper pipeline"
    )
    probe_algo2_parser.add_argument("--embedding-model", required=True)
    probe_algo2_parser.add_argument("--seed-label", action="append", required=True)
    probe_algo2_parser.add_argument("--convergence-threshold", type=float, required=True)
    probe_algo2_parser.add_argument("--output-dir", required=True)
    probe_algo2_parser.add_argument("--resume", action="store_true")

    probe_algo3_parser = probe_subparsers.add_parser("algo3")
    probe_algo3_parser.add_argument("--run-name", required=True)
    probe_algo3_parser.add_argument("--model", required=True)
    probe_algo3_parser.add_argument(
        "--provider", default="mistral", choices=["mistral", "anthropic"],
        help="DEBUG: 'anthropic' is for research/debugging only, not part of paper pipeline"
    )
    probe_algo3_parser.add_argument("--source-label", action="append", required=True)
    probe_algo3_parser.add_argument("--target-label", action="append", required=True)
    probe_algo3_parser.add_argument("--child-count", type=int, required=True)
    probe_algo3_parser.add_argument("--max-depth", type=int, required=True)
    probe_algo3_parser.add_argument("--output-dir", required=True)
    probe_algo3_parser.add_argument("--resume", action="store_true")

    summary_parser = analyze_subparsers.add_parser("summary")
    summary_parser.add_argument("--input", action="append", required=True)
    summary_parser.add_argument("--group-by", action="append", required=True)
    summary_parser.add_argument("--metric", action="append", required=True)
    summary_parser.add_argument("--output", required=True)

    failures_parser = analyze_subparsers.add_parser("failures")
    failures_parser.add_argument("--input", action="append", required=True)
    failures_parser.add_argument("--result-column", required=True)
    failures_parser.add_argument("--output", required=True)

    stability_parser = analyze_subparsers.add_parser("stability")
    stability_parser.add_argument("--input", action="append", required=True)
    stability_parser.add_argument("--group-by", action="append", required=True)
    stability_parser.add_argument("--metric", action="append", required=True)
    stability_parser.add_argument("--output", required=True)

    hypothesis_parser = analyze_subparsers.add_parser("hypothesis")
    hypothesis_parser.add_argument("--input", action="append", required=True)
    hypothesis_parser.add_argument("--factor", required=True)
    hypothesis_parser.add_argument("--pair-by", action="append", required=True)
    hypothesis_parser.add_argument("--metric", action="append", required=True)
    hypothesis_parser.add_argument("--output", required=True)

    figures_parser = analyze_subparsers.add_parser("figures")
    figures_parser.add_argument("--input", action="append", required=True)
    figures_parser.add_argument("--id-column", action="append", required=True)
    figures_parser.add_argument("--metric", action="append", required=True)
    figures_parser.add_argument("--output", required=True)

    baseline_comparison_parser = analyze_subparsers.add_parser("baseline-comparison")
    baseline_comparison_parser.add_argument("--input", action="append", required=True)
    baseline_comparison_parser.add_argument("--baseline-input", action="append", required=True)
    baseline_comparison_parser.add_argument("--metric", action="append", required=True)
    baseline_comparison_parser.add_argument("--output", required=True)

    paper_alignment_parser = audit_subparsers.add_parser("paper-alignment")
    paper_alignment_parser.add_argument("--json", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        return handle_doctor(args)
    if args.command == "analyze":
        return handle_analyze(args)
    if args.command == "audit":
        return handle_audit(args)
    if args.command == "eval":
        return handle_eval(args)
    if args.command == "baseline":
        return handle_baseline(args)
    if args.command == "factorial":
        return handle_factorial(args)
    if args.command == "verify":
        return handle_verify(args)
    if args.command == "generate":
        return handle_generate(args)
    if args.command == "probe":
        return handle_probe(args)

    parser.error("unsupported command")
    return 2


def run() -> None:
    raise SystemExit(main())
