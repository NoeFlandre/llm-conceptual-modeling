import json
import os
import sys
from argparse import Namespace
from pathlib import Path

from llm_conceptual_modeling.algo1.mistral import (
    Method1PromptConfig,
)
from llm_conceptual_modeling.algo1.mistral import (
    MistralChatClient as Algo1MistralChatClient,
)
from llm_conceptual_modeling.algo1.probe import Algo1ProbeSpec, run_algo1_probe
from llm_conceptual_modeling.algo2.embeddings import MistralEmbeddingClient as Algo2EmbeddingClient
from llm_conceptual_modeling.algo2.mistral import (
    Method2PromptConfig,
)
from llm_conceptual_modeling.algo2.mistral import (
    MistralChatClient as Algo2MistralChatClient,
)
from llm_conceptual_modeling.algo2.probe import Algo2ProbeSpec, run_algo2_probe
from llm_conceptual_modeling.algo3.mistral import (
    Method3PromptConfig,
)
from llm_conceptual_modeling.algo3.mistral import (
    MistralChatClient as Algo3MistralChatClient,
)
from llm_conceptual_modeling.algo3.probe import Algo3ProbeSpec, run_algo3_probe
from llm_conceptual_modeling.common.anthropic import AnthropicChatClient
from llm_conceptual_modeling.generation import emit_json

# Backward-compatible aliases for test monkeypatching
Algo1ChatClient = Algo1MistralChatClient
Algo2ChatClient = Algo2MistralChatClient
Algo3ChatClient = Algo3MistralChatClient

Edge = tuple[str, str]


def handle_probe(args: Namespace) -> int:
    try:
        if args.provider == "anthropic":
            api_key = _require_api_key("ANTHROPIC_API_KEY")
        else:
            api_key = _require_api_key("MISTRAL_API_KEY")

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
    subgraph1 = _parse_edge_list(args.subgraph1_edge)
    subgraph2 = _parse_edge_list(args.subgraph2_edge)
    output_dir = Path(args.output_dir)
    spec = Algo1ProbeSpec(
        run_name=args.run_name,
        model=args.model,
        subgraph1=subgraph1,
        subgraph2=subgraph2,
        prompt_config=Method1PromptConfig(
            use_adjacency_notation=False,
            use_array_representation=False,
            include_explanation=False,
            include_example=False,
            include_counterexample=False,
        ),
        output_dir=output_dir,
        resume=args.resume,
    )
    if args.provider == "anthropic":
        chat_client = AnthropicChatClient(
            api_key=api_key,
            model=args.model,
        )
    else:
        chat_client = Algo1ChatClient(
            api_key=api_key,
            model=args.model,
        )
    summary = run_algo1_probe(
        spec=spec,
        chat_client=chat_client,
    )
    emit_json(summary)
    return 0


def _handle_algo2_probe(args: Namespace, *, api_key: str) -> int:
    output_dir = Path(args.output_dir)
    spec = Algo2ProbeSpec(
        run_name=args.run_name,
        model=args.model,
        seed_labels=args.seed_label,
        subgraph1=[],
        subgraph2=[],
        prompt_config=Method2PromptConfig(
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
    if args.provider == "anthropic":
        chat_client = AnthropicChatClient(
            api_key=api_key,
            model=args.model,
        )
    else:
        chat_client = Algo2ChatClient(
            api_key=api_key,
            model=args.model,
        )
    embedding_client = Algo2EmbeddingClient(
        api_key=api_key,
        model=args.embedding_model,
    )
    summary = run_algo2_probe(
        spec=spec,
        chat_client=chat_client,
        embedding_client=embedding_client,
    )
    emit_json(summary)
    return 0


def _handle_algo3_probe(args: Namespace, *, api_key: str) -> int:
    output_dir = Path(args.output_dir)
    spec = Algo3ProbeSpec(
        run_name=args.run_name,
        model=args.model,
        source_labels=args.source_label,
        target_labels=args.target_label,
        prompt_config=Method3PromptConfig(
            include_example=False,
            include_counterexample=False,
        ),
        child_count=args.child_count,
        max_depth=args.max_depth,
        output_dir=output_dir,
        resume=args.resume,
    )
    if args.provider == "anthropic":
        chat_client = AnthropicChatClient(
            api_key=api_key,
            model=args.model,
        )
    else:
        chat_client = Algo3ChatClient(
            api_key=api_key,
            model=args.model,
        )
    summary = run_algo3_probe(
        spec=spec,
        chat_client=chat_client,
    )
    emit_json(summary)
    return 0


def _require_api_key(environment_variable: str) -> str:
    api_key = os.environ.get(environment_variable)
    if api_key:
        return api_key

    raise ValueError(f"Missing required environment variable: {environment_variable}")


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
