from argparse import Namespace

from llm_conceptual_modeling.verification import (
    emit_json,
    run_full_verification,
    run_legacy_parity_verification,
)


def handle_verify(args: Namespace) -> int:
    if args.verify_target == "legacy-parity":
        emit_json(run_legacy_parity_verification())
        return 0
    if args.verify_target == "all":
        emit_json(run_full_verification())
        return 0
    raise ValueError(f"Unsupported verify target: {args.verify_target}")
