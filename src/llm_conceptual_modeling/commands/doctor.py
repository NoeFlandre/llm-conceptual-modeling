from argparse import Namespace

from llm_conceptual_modeling.verification import build_doctor_report, emit_json


def handle_doctor(args: Namespace) -> int:
    emit_json(build_doctor_report())
    return 0
