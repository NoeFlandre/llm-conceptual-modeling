import os
import sys
from argparse import Namespace
from pathlib import Path

from llm_conceptual_modeling.algo1.experiment import (
    build_algo1_experiment_specs,
    run_algo1_experiment,
)
from llm_conceptual_modeling.algo1.mistral import MistralChatClient as Algo1ChatClient
from llm_conceptual_modeling.generation import (
    build_generation_stub_payload,
    emit_json,
)


def handle_generate(args: Namespace) -> int:
    if _should_execute_algo1_experiment(args):
        return _handle_algo1_execution(args)

    emit_json(
        build_generation_stub_payload(
            args.algorithm,
            fixture_only=args.fixture_only,
        )
    )
    return 0


def _should_execute_algo1_experiment(args: Namespace) -> bool:
    if args.algorithm != "algo1":
        return False

    required_values_present = bool(args.model and args.pair and args.output_root)
    return required_values_present


def _handle_algo1_execution(args: Namespace) -> int:
    try:
        api_key = _require_api_key("MISTRAL_API_KEY")
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    output_root = Path(args.output_root)
    specs = build_algo1_experiment_specs(
        pair_name=args.pair,
        output_root=output_root,
        replications=args.replications,
    )
    chat_client = Algo1ChatClient(
        api_key=api_key,
        model=args.model,
    )
    summary_records = run_algo1_experiment(
        specs=specs,
        chat_client=chat_client,
    )
    payload = {
        "algorithm": "algo1",
        "mode": "executed-experiment",
        "pair": args.pair,
        "model": args.model,
        "replications": args.replications,
        "result_count": len(summary_records),
        "results": summary_records,
    }
    emit_json(payload)
    return 0


def _require_api_key(environment_variable: str) -> str:
    api_key = os.environ.get(environment_variable)
    if api_key:
        return api_key

    raise ValueError(f"Missing required environment variable: {environment_variable}")
