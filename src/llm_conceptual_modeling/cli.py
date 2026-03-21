import argparse
from collections.abc import Sequence

from llm_conceptual_modeling.algo1.evaluation import evaluate_results_file
from llm_conceptual_modeling.algo1.factorial import (
    run_factorial_analysis as run_algo1_factorial_analysis,
)
from llm_conceptual_modeling.algo2.evaluation import (
    evaluate_results_file as evaluate_algo2_results_file,
)
from llm_conceptual_modeling.algo2.factorial import (
    run_factorial_analysis as run_algo2_factorial_analysis,
)
from llm_conceptual_modeling.algo3.evaluation import (
    evaluate_results_file as evaluate_algo3_results_file,
)
from llm_conceptual_modeling.algo3.factorial import (
    run_factorial_analysis as run_algo3_factorial_analysis,
)
from llm_conceptual_modeling.generation import (
    build_generation_stub_payload,
)
from llm_conceptual_modeling.generation import (
    emit_json as emit_generation_json,
)
from llm_conceptual_modeling.verification import (
    build_doctor_report,
    run_legacy_parity_verification,
)
from llm_conceptual_modeling.verification import (
    emit_json as emit_verification_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lcm")
    subparsers = parser.add_subparsers(dest="command", required=True)

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

    legacy_parity_parser = verify_subparsers.add_parser("legacy-parity")
    legacy_parity_parser.add_argument("--json", action="store_true")

    for algorithm in ("algo1", "algo2", "algo3"):
        generate_algorithm_parser = generate_subparsers.add_parser(algorithm)
        generate_algorithm_parser.add_argument("--fixture-only", action="store_true")
        generate_algorithm_parser.add_argument("--json", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "eval" and args.algorithm == "algo1":
        evaluate_results_file(args.input, args.output)
        return 0
    if args.command == "eval" and args.algorithm == "algo2":
        evaluate_algo2_results_file(args.input, args.output)
        return 0
    if args.command == "eval" and args.algorithm == "algo3":
        evaluate_algo3_results_file(args.input, args.output)
        return 0
    if args.command == "factorial" and args.algorithm == "algo1":
        run_algo1_factorial_analysis(args.input, args.output)
        return 0
    if args.command == "factorial" and args.algorithm == "algo2":
        run_algo2_factorial_analysis(args.input, args.output)
        return 0
    if args.command == "factorial" and args.algorithm == "algo3":
        run_algo3_factorial_analysis(args.input, args.output)
        return 0
    if args.command == "doctor":
        emit_verification_json(build_doctor_report())
        return 0
    if args.command == "verify" and args.verify_target == "legacy-parity":
        emit_verification_json(run_legacy_parity_verification())
        return 0
    if args.command == "generate":
        emit_generation_json(
            build_generation_stub_payload(
                args.algorithm,
                fixture_only=args.fixture_only,
            )
        )
        return 0

    parser.error("unsupported command")
    return 2


def run() -> None:
    raise SystemExit(main())
