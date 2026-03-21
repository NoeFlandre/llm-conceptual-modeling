from argparse import Namespace

from llm_conceptual_modeling.generation import emit_json
from llm_conceptual_modeling.verification import build_paper_alignment_report


def handle_audit(args: Namespace) -> int:
    if args.audit_target == "paper-alignment":
        emit_json(build_paper_alignment_report())
        return 0

    raise ValueError(f"Unsupported audit target: {args.audit_target}")
