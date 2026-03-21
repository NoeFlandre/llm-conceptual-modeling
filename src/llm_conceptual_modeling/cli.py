import argparse
from collections.abc import Sequence

from llm_conceptual_modeling.commands.doctor import handle_doctor
from llm_conceptual_modeling.commands.eval import handle_eval
from llm_conceptual_modeling.commands.factorial import handle_factorial
from llm_conceptual_modeling.commands.generate import handle_generate
from llm_conceptual_modeling.commands.verify import handle_verify


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

    verify_all_parser = verify_subparsers.add_parser("all")
    verify_all_parser.add_argument("--json", action="store_true")

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

    if args.command == "doctor":
        return handle_doctor(args)
    if args.command == "eval":
        return handle_eval(args)
    if args.command == "factorial":
        return handle_factorial(args)
    if args.command == "verify":
        return handle_verify(args)
    if args.command == "generate":
        return handle_generate(args)

    parser.error("unsupported command")
    return 2


def run() -> None:
    raise SystemExit(main())
