from argparse import Namespace

from llm_conceptual_modeling.verification import (
    emit_json,
    run_full_verification,
    run_legacy_parity_verification,
)


def handle_verify(args: Namespace) -> int:
    if args.verify_target == "legacy-parity":
        report = run_legacy_parity_verification()
        emit_json(report)
        return 0 if report["status"] == "ok" else 1
    if args.verify_target == "all":
        report = run_full_verification()
        emit_json(report)
        return 0 if report["status"] == "ok" else 1
    raise ValueError(f"Unsupported verify target: {args.verify_target}")
