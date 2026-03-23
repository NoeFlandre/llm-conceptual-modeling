import json
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

from llm_conceptual_modeling.commands._provider_utils import (
    build_chat_client,
    resolve_provider_api_key,
)
from llm_conceptual_modeling.generation import emit_json

# Backward-compatible aliases for test monkeypatching.
Algo1ChatClient = None
Algo2ChatClient = None
Algo3ChatClient = None
Algo2EmbeddingClient = None
Edge = tuple[str, str]


@dataclass(frozen=True)
class Method1PromptConfig:
    use_adjacency_notation: bool
    use_array_representation: bool
    include_explanation: bool
    include_example: bool
    include_counterexample: bool


@dataclass(frozen=True)
class Method2PromptConfig:
    use_adjacency_notation: bool
    use_array_representation: bool
    include_explanation: bool
    include_example: bool
    include_counterexample: bool


@dataclass(frozen=True)
class Method3PromptConfig:
    include_example: bool
    include_counterexample: bool


@dataclass(frozen=True)
class Algo1ProbeSpec:
    run_name: str
    model: str
    subgraph1: list[Edge]
    subgraph2: list[Edge]
    prompt_config: Method1PromptConfig
    output_dir: Path
    resume: bool = False


@dataclass(frozen=True)
class Algo2ProbeSpec:
    run_name: str
    model: str
    seed_labels: list[str]
    subgraph1: list[Edge]
    subgraph2: list[Edge]
    prompt_config: Method2PromptConfig
    convergence_threshold: float
    output_dir: Path
    resume: bool = False


@dataclass(frozen=True)
class Algo3ProbeSpec:
    run_name: str
    model: str
    source_labels: list[str]
    target_labels: list[str]
    prompt_config: Method3PromptConfig
    child_count: int
    max_depth: int
    output_dir: Path
    resume: bool = False


run_algo1_probe = None
run_algo2_probe = None
run_algo3_probe = None


def handle_probe(args: Namespace) -> int:
    try:
        api_key = resolve_provider_api_key(args.provider)

        if args.algorithm == "algo1":
            return _handle_algo1_probe(args, api_key=api_key)
        if args.algorithm == "algo2":
            return _handle_algo2_probe(args, api_key=api_key)
        if args.algorithm == "algo3":
            return _handle_algo3_probe(args, api_key=api_key)

        raise ValueError(f"Unsupported probe algorithm: {args.algorithm}")
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1


def _handle_algo1_probe(args: Namespace, *, api_key: str) -> int:
    (
        probe_spec_class,
        prompt_config_class,
        probe_runner,
        chat_client_class,
    ) = _load_algo1_runtime_symbols()
    subgraph1 = _parse_edge_list(args.subgraph1_edge)
    subgraph2 = _parse_edge_list(args.subgraph2_edge)
    output_dir = Path(args.output_dir)
    spec = probe_spec_class(
        run_name=args.run_name,
        model=args.model,
        subgraph1=subgraph1,
        subgraph2=subgraph2,
        prompt_config=prompt_config_class(
            use_adjacency_notation=False,
            use_array_representation=False,
            include_explanation=False,
            include_example=False,
            include_counterexample=False,
        ),
        output_dir=output_dir,
        resume=args.resume,
    )
    chat_client = build_chat_client(
        provider=args.provider,
        api_key=api_key,
        model=args.model,
        mistral_chat_client_class=chat_client_class,
    )
    summary = probe_runner(
        spec=spec,
        chat_client=chat_client,
    )
    emit_json(summary)
    return 0


def _handle_algo2_probe(args: Namespace, *, api_key: str) -> int:
    (
        probe_spec_class,
        prompt_config_class,
        probe_runner,
        chat_client_class,
        embedding_client_class,
    ) = _load_algo2_runtime_symbols()
    output_dir = Path(args.output_dir)
    spec = probe_spec_class(
        run_name=args.run_name,
        model=args.model,
        seed_labels=args.seed_label,
        subgraph1=[],
        subgraph2=[],
        prompt_config=prompt_config_class(
            use_adjacency_notation=False,
            use_array_representation=False,
            include_explanation=False,
            include_example=False,
            include_counterexample=False,
        ),
        convergence_threshold=args.convergence_threshold,
        output_dir=output_dir,
        resume=args.resume,
    )
    chat_client = build_chat_client(
        provider=args.provider,
        api_key=api_key,
        model=args.model,
        mistral_chat_client_class=chat_client_class,
    )
    embedding_client = embedding_client_class(
        api_key=api_key,
        model=args.embedding_model,
    )
    summary = probe_runner(
        spec=spec,
        chat_client=chat_client,
        embedding_client=embedding_client,
    )
    emit_json(summary)
    return 0


def _handle_algo3_probe(args: Namespace, *, api_key: str) -> int:
    (
        probe_spec_class,
        prompt_config_class,
        probe_runner,
        chat_client_class,
    ) = _load_algo3_runtime_symbols()
    output_dir = Path(args.output_dir)
    spec = probe_spec_class(
        run_name=args.run_name,
        model=args.model,
        source_labels=args.source_label,
        target_labels=args.target_label,
        prompt_config=prompt_config_class(
            include_example=False,
            include_counterexample=False,
        ),
        child_count=args.child_count,
        max_depth=args.max_depth,
        output_dir=output_dir,
        resume=args.resume,
    )
    chat_client = build_chat_client(
        provider=args.provider,
        api_key=api_key,
        model=args.model,
        mistral_chat_client_class=chat_client_class,
    )
    summary = probe_runner(
        spec=spec,
        chat_client=chat_client,
    )
    emit_json(summary)
    return 0


def _load_algo1_runtime_symbols():
    global Algo1ChatClient, Method1PromptConfig, Algo1ProbeSpec, run_algo1_probe

    if (
        Algo1ChatClient is None
        or Method1PromptConfig is None
        or Algo1ProbeSpec is None
        or run_algo1_probe is None
    ):
        from llm_conceptual_modeling.algo1.mistral import (
            Method1PromptConfig as loaded_prompt_config,
        )
        from llm_conceptual_modeling.algo1.mistral import (
            MistralChatClient as loaded_chat_client,
        )
        from llm_conceptual_modeling.algo1.probe import (
            Algo1ProbeSpec as loaded_probe_spec,
        )
        from llm_conceptual_modeling.algo1.probe import (
            run_algo1_probe as loaded_probe_runner,
        )

        if Algo1ChatClient is None:
            Algo1ChatClient = loaded_chat_client
        if Method1PromptConfig is None:
            Method1PromptConfig = loaded_prompt_config
        if Algo1ProbeSpec is None:
            Algo1ProbeSpec = loaded_probe_spec
        if run_algo1_probe is None:
            run_algo1_probe = loaded_probe_runner

    return Algo1ProbeSpec, Method1PromptConfig, run_algo1_probe, Algo1ChatClient


def _load_algo2_runtime_symbols():
    global Algo2ChatClient, Algo2EmbeddingClient, Method2PromptConfig, Algo2ProbeSpec, run_algo2_probe

    if (
        Algo2ChatClient is None
        or Algo2EmbeddingClient is None
        or Method2PromptConfig is None
        or Algo2ProbeSpec is None
        or run_algo2_probe is None
    ):
        from llm_conceptual_modeling.algo2.embeddings import (
            MistralEmbeddingClient as loaded_embedding_client,
        )
        from llm_conceptual_modeling.algo2.mistral import (
            Method2PromptConfig as loaded_prompt_config,
        )
        from llm_conceptual_modeling.algo2.mistral import (
            MistralChatClient as loaded_chat_client,
        )
        from llm_conceptual_modeling.algo2.probe import (
            Algo2ProbeSpec as loaded_probe_spec,
        )
        from llm_conceptual_modeling.algo2.probe import (
            run_algo2_probe as loaded_probe_runner,
        )

        if Algo2ChatClient is None:
            Algo2ChatClient = loaded_chat_client
        if Algo2EmbeddingClient is None:
            Algo2EmbeddingClient = loaded_embedding_client
        if Method2PromptConfig is None:
            Method2PromptConfig = loaded_prompt_config
        if Algo2ProbeSpec is None:
            Algo2ProbeSpec = loaded_probe_spec
        if run_algo2_probe is None:
            run_algo2_probe = loaded_probe_runner

    return (
        Algo2ProbeSpec,
        Method2PromptConfig,
        run_algo2_probe,
        Algo2ChatClient,
        Algo2EmbeddingClient,
    )


def _load_algo3_runtime_symbols():
    global Algo3ChatClient, Method3PromptConfig, Algo3ProbeSpec, run_algo3_probe

    if (
        Algo3ChatClient is None
        or Method3PromptConfig is None
        or Algo3ProbeSpec is None
        or run_algo3_probe is None
    ):
        from llm_conceptual_modeling.algo3.mistral import (
            Method3PromptConfig as loaded_prompt_config,
        )
        from llm_conceptual_modeling.algo3.mistral import (
            MistralChatClient as loaded_chat_client,
        )
        from llm_conceptual_modeling.algo3.probe import (
            Algo3ProbeSpec as loaded_probe_spec,
        )
        from llm_conceptual_modeling.algo3.probe import (
            run_algo3_probe as loaded_probe_runner,
        )

        if Algo3ChatClient is None:
            Algo3ChatClient = loaded_chat_client
        if Method3PromptConfig is None:
            Method3PromptConfig = loaded_prompt_config
        if Algo3ProbeSpec is None:
            Algo3ProbeSpec = loaded_probe_spec
        if run_algo3_probe is None:
            run_algo3_probe = loaded_probe_runner

    return Algo3ProbeSpec, Method3PromptConfig, run_algo3_probe, Algo3ChatClient


def _parse_edge_list(edge_values: list[str]) -> list[Edge]:
    parsed_edges: list[Edge] = []

    for edge_value in edge_values:
        parsed_edge = _parse_edge_value(edge_value)
        parsed_edges.append(parsed_edge)

    return parsed_edges


def _parse_edge_value(edge_value: str) -> Edge:
    parsed_value = json.loads(edge_value)
    if not isinstance(parsed_value, list) or len(parsed_value) != 2:
        raise ValueError(f"Invalid edge JSON value: {edge_value}")

    source_text = str(parsed_value[0])
    target_text = str(parsed_value[1])
    return (source_text, target_text)
