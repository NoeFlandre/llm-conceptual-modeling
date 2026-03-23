import sys
from argparse import Namespace
from pathlib import Path

from llm_conceptual_modeling.commands._provider_utils import (
    build_chat_client,
    build_embedding_client,
    resolve_provider_api_key,
)
from llm_conceptual_modeling.common.model_catalog import resolve_model_alias
from llm_conceptual_modeling.generation import (
    build_generation_stub_payload,
    emit_json,
)

# Backward-compatible aliases for test monkeypatching.
Algo1ChatClient = None
Algo2ChatClient = None
Algo3ChatClient = None
build_algo1_experiment_specs = None
run_algo1_experiment = None
build_algo2_experiment_specs = None
run_algo2_experiment = None
build_algo3_experiment_specs = None
run_algo3_experiment = None
Algo2EmbeddingClient = None


def handle_generate(args: Namespace) -> int:
    if _should_execute_algo1_experiment(args):
        return _handle_algo1_execution(args)
    if _should_execute_algo2_experiment(args):
        return _handle_algo2_execution(args)
    if _should_execute_algo3_experiment(args):
        return _handle_algo3_execution(args)

    emit_json(
        build_generation_stub_payload(
            args.algorithm,
            fixture_only=args.fixture_only,
        )
    )
    return 0


def _should_execute_algo1_experiment(args: Namespace) -> bool:
    return args.algorithm == "algo1" and bool(args.model and args.pair and args.output_root)


def _should_execute_algo2_experiment(args: Namespace) -> bool:
    return args.algorithm == "algo2" and bool(
        args.model
        and args.embedding_model
        and args.embedding_provider
        and args.pair
        and args.output_root
    )


def _should_execute_algo3_experiment(args: Namespace) -> bool:
    return args.algorithm == "algo3" and bool(args.model and args.pair and args.output_root)


def _handle_algo1_execution(args: Namespace) -> int:
    try:
        api_key = resolve_provider_api_key(args.provider)
        resolved_model = resolve_model_alias(
            provider=args.provider,
            model=args.model,
            role="chat",
        )
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    build_specs, run_experiment, chat_client_class = _load_algo1_runtime_symbols()
    output_root = Path(args.output_root)
    specs = build_specs(
        pair_name=args.pair,
        model=resolved_model,
        provider=args.provider,
        output_root=output_root,
        replications=args.replications,
        resume=args.resume,
    )
    chat_client = build_chat_client(
        provider=args.provider,
        api_key=api_key,
        model=resolved_model,
        mistral_chat_client_class=chat_client_class,
    )
    summary_records = run_experiment(
        specs=specs,
        chat_client=chat_client,
    )
    payload = {
        "algorithm": "algo1",
        "mode": "executed-experiment",
        "pair": args.pair,
        "model": resolved_model,
        "provider": args.provider,
        "replications": args.replications,
        "result_count": len(summary_records),
        "results": summary_records,
    }
    emit_json(payload)
    return 0


def _handle_algo2_execution(args: Namespace) -> int:
    try:
        api_key = resolve_provider_api_key(args.provider)
        embedding_api_key = resolve_provider_api_key(args.embedding_provider)
        resolved_model = resolve_model_alias(
            provider=args.provider,
            model=args.model,
            role="chat",
        )
        resolved_embedding_model = resolve_model_alias(
            provider=args.embedding_provider,
            model=args.embedding_model,
            role="embedding",
        )
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    (
        build_specs,
        run_experiment,
        chat_client_class,
        embedding_client_class,
    ) = _load_algo2_runtime_symbols()
    output_root = Path(args.output_root)
    specs = build_specs(
        pair_name=args.pair,
        model=resolved_model,
        provider=args.provider,
        embedding_provider=args.embedding_provider,
        embedding_model=resolved_embedding_model,
        output_root=output_root,
        replications=args.replications,
        resume=args.resume,
    )
    chat_client = build_chat_client(
        provider=args.provider,
        api_key=api_key,
        model=resolved_model,
        mistral_chat_client_class=chat_client_class,
    )
    embedding_client = build_embedding_client(
        provider=args.embedding_provider,
        api_key=embedding_api_key,
        model=resolved_embedding_model,
        mistral_embedding_client_class=embedding_client_class,
    )
    summary_records = run_experiment(
        specs=specs,
        chat_client=chat_client,
        embedding_client=embedding_client,
    )
    payload = {
        "algorithm": "algo2",
        "mode": "executed-experiment",
        "pair": args.pair,
        "model": resolved_model,
        "provider": args.provider,
        "embedding_provider": args.embedding_provider,
        "embedding_model": resolved_embedding_model,
        "replications": args.replications,
        "result_count": len(summary_records),
        "results": summary_records,
    }
    emit_json(payload)
    return 0


def _handle_algo3_execution(args: Namespace) -> int:
    try:
        api_key = resolve_provider_api_key(args.provider)
        resolved_model = resolve_model_alias(
            provider=args.provider,
            model=args.model,
            role="chat",
        )
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    build_specs, run_experiment, chat_client_class = _load_algo3_runtime_symbols()
    output_root = Path(args.output_root)
    specs = build_specs(
        pair_name=args.pair,
        model=resolved_model,
        provider=args.provider,
        output_root=output_root,
        replications=args.replications,
        resume=args.resume,
    )
    chat_client = build_chat_client(
        provider=args.provider,
        api_key=api_key,
        model=resolved_model,
        mistral_chat_client_class=chat_client_class,
    )
    summary_records = run_experiment(
        specs=specs,
        chat_client=chat_client,
    )
    payload = {
        "algorithm": "algo3",
        "mode": "executed-experiment",
        "pair": args.pair,
        "model": resolved_model,
        "provider": args.provider,
        "replications": args.replications,
        "result_count": len(summary_records),
        "results": summary_records,
    }
    emit_json(payload)
    return 0


def _load_algo1_runtime_symbols():
    global Algo1ChatClient, build_algo1_experiment_specs, run_algo1_experiment

    if (
        Algo1ChatClient is None
        or build_algo1_experiment_specs is None
        or run_algo1_experiment is None
    ):
        from llm_conceptual_modeling.algo1.experiment import (
            build_algo1_experiment_specs as loaded_build_specs,
        )
        from llm_conceptual_modeling.algo1.experiment import (
            run_algo1_experiment as loaded_run_experiment,
        )
        from llm_conceptual_modeling.algo1.mistral import (
            MistralChatClient as loaded_chat_client,
        )

        if Algo1ChatClient is None:
            Algo1ChatClient = loaded_chat_client
        if build_algo1_experiment_specs is None:
            build_algo1_experiment_specs = loaded_build_specs
        if run_algo1_experiment is None:
            run_algo1_experiment = loaded_run_experiment

    return build_algo1_experiment_specs, run_algo1_experiment, Algo1ChatClient


def _load_algo2_runtime_symbols():
    global Algo2ChatClient, Algo2EmbeddingClient, build_algo2_experiment_specs, run_algo2_experiment

    if (
        Algo2ChatClient is None
        or Algo2EmbeddingClient is None
        or build_algo2_experiment_specs is None
        or run_algo2_experiment is None
    ):
        from llm_conceptual_modeling.algo2.embeddings import (
            MistralEmbeddingClient as loaded_embedding_client,
        )
        from llm_conceptual_modeling.algo2.experiment import (
            build_algo2_experiment_specs as loaded_build_specs,
        )
        from llm_conceptual_modeling.algo2.experiment import (
            run_algo2_experiment as loaded_run_experiment,
        )
        from llm_conceptual_modeling.algo2.mistral import (
            MistralChatClient as loaded_chat_client,
        )

        if Algo2ChatClient is None:
            Algo2ChatClient = loaded_chat_client
        if Algo2EmbeddingClient is None:
            Algo2EmbeddingClient = loaded_embedding_client
        if build_algo2_experiment_specs is None:
            build_algo2_experiment_specs = loaded_build_specs
        if run_algo2_experiment is None:
            run_algo2_experiment = loaded_run_experiment

    return (
        build_algo2_experiment_specs,
        run_algo2_experiment,
        Algo2ChatClient,
        Algo2EmbeddingClient,
    )


def _load_algo3_runtime_symbols():
    global Algo3ChatClient, build_algo3_experiment_specs, run_algo3_experiment

    if (
        Algo3ChatClient is None
        or build_algo3_experiment_specs is None
        or run_algo3_experiment is None
    ):
        from llm_conceptual_modeling.algo3.experiment import (
            build_algo3_experiment_specs as loaded_build_specs,
        )
        from llm_conceptual_modeling.algo3.experiment import (
            run_algo3_experiment as loaded_run_experiment,
        )
        from llm_conceptual_modeling.algo3.mistral import (
            MistralChatClient as loaded_chat_client,
        )

        if Algo3ChatClient is None:
            Algo3ChatClient = loaded_chat_client
        if build_algo3_experiment_specs is None:
            build_algo3_experiment_specs = loaded_build_specs
        if run_algo3_experiment is None:
            run_algo3_experiment = loaded_run_experiment

    return build_algo3_experiment_specs, run_algo3_experiment, Algo3ChatClient
