from argparse import Namespace

from llm_conceptual_modeling.generation import (
    build_generation_stub_payload,
    emit_json,
)


def handle_generate(args: Namespace) -> int:
    emit_json(
        build_generation_stub_payload(
            args.algorithm,
            fixture_only=args.fixture_only,
            provider=args.provider,
            config_path=getattr(args, "config", None),
        )
    )
    return 0
