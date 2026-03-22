import json
from dataclasses import dataclass
from typing import Any, cast as typing_cast, Protocol
from llm_conceptual_modeling.algo1.cove import apply_cove_verification, build_cove_prompt

from mistralai.client import Mistral



@dataclass(frozen=True)
class Method1PromptConfig:
    use_adjacency_notation: bool
    use_array_representation: bool
    include_explanation: bool
    include_example: bool
    include_counterexample: bool


class MistralChatClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        sdk_client: Any | None = None,
    ) -> None:
        self._model = model
        self._sdk_client = sdk_client or Mistral(api_key=api_key)

    def complete_json(
        self,
        *,
        prompt: str,
        schema_name: str,
        schema: dict[str, object],
    ) -> dict[str, object]:
        response = call_with_retry(
            operation=lambda: self._sdk_client.chat.complete(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": schema,
                    },
                },
            ),
            operation_name="mistral chat completion",
            max_attempts=8,
            initial_delay_seconds=2.0,
            max_delay_seconds=30.0,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Mistral returned an empty chat completion content")
        parsed_content = json.loads(content)
        return parsed_content


class ChatCompletionClient(Protocol):
    def complete_json(
        self,
        *,
        prompt: str,
        schema_name: str,
        schema: dict[str, object],
    ) -> dict[str, object]: ...


def build_direct_edge_prompt(
    *,
    subgraph1: list[Edge],
    subgraph2: list[Edge],
    prompt_config: Method1PromptConfig | None = None,
) -> str:
    resolved_prompt_config = prompt_config or Method1PromptConfig(
        use_adjacency_notation=False,
        use_array_representation=False,
        include_explanation=False,
        include_example=False,
        include_counterexample=False,
    )

    prompt_sections: list[str] = []
    prompt_sections.append("You are a helpful assistant who understands Knowledge Maps.")

    if resolved_prompt_config.include_explanation:
        prompt_sections.append(
            "A knowledge map is a network consisting of nodes and edges. "
            "Nodes must have a clear meaning, such that we can interpret having "
            "'more' or 'less' of a node. Edges represent the existence of a direct "
            "relation between two nodes."
        )

    notation_section = _build_notation_section(
        use_adjacency_notation=resolved_prompt_config.use_adjacency_notation,
        use_array_representation=resolved_prompt_config.use_array_representation,
    )
    prompt_sections.append(notation_section)

    if resolved_prompt_config.include_example:
        prompt_sections.append(_build_example_section())

    if resolved_prompt_config.include_counterexample:
        prompt_sections.append(_build_counterexample_section())

    formatted_subgraph1 = _format_knowledge_map(
        subgraph1,
        prompt_config=resolved_prompt_config,
    )
    formatted_subgraph2 = _format_knowledge_map(
        subgraph2,
        prompt_config=resolved_prompt_config,
    )
    prompt_sections.append(
        "You will get two inputs: "
        f"Knowledge map 1: {formatted_subgraph1} "
        f"Knowledge map 2: {formatted_subgraph2}"
    )
    prompt_sections.append(
        "Your task is to recommend more links between the two maps. "
        "These links can use new nodes. "
        "Do not suggest links that are already in the maps. "
        "Do not suggest links between nodes of the same map. "
        "Return a JSON object with a top-level 'edges' array. "
        "Each edge must be an object with 'source' and 'target' string fields."
    )
    prompt_sections.append(
        "Your output must only be the list of proposed edges. "
        "Do not repeat any instructions I have given you and do not add "
        "unnecessary words or phrases."
    )
    prompt = " ".join(prompt_sections)
    return prompt


def _build_notation_section(
    *,
    use_adjacency_notation: bool,
    use_array_representation: bool,
) -> str:
    if use_adjacency_notation:
        if use_array_representation:
            return (
                "The knowledge map is encoded using a list of nodes and an associated "
                "adjacency matrix. The adjacency matrix is an n*n square matrix that "
                "represents whether each edge exists."
            )

        return (
            "The knowledge map is encoded using tags for nodes and an associated "
            "adjacency matrix."
        )

    if use_array_representation:
        return "The knowledge map is encoded using arrays of directed edges."

    return "The knowledge map is encoded using tags that store directed edges."


def _build_example_section() -> str:
    return (
        "Here is an example of a desired output for your task. "
        "In knowledge map 1, we have the list of nodes ['capacity to hire', "
        "'bad employees', 'good reputation'] and the associated adjacency matrix "
        "[[0,1,0],[0,0,1],[1,0,0]]. In knowledge map 2, we have the list of nodes "
        "['work motivation', 'productivity', 'financial growth'] and the associated "
        "adjacency matrix [[0,1,0],[0,0,1],[0,0,0]]. In this example, you could "
        "recommend 3 new links."
    )


def _build_counterexample_section() -> str:
    return (
        "Here is an example of a bad output that we do not want to see. "
        "A bad output would be: [('moon', 'bad employees')]. "
        "The proposed link does not represent a true causal relationship."
    )


def _format_knowledge_map(
    edges: list[Edge],
    *,
    prompt_config: Method1PromptConfig,
) -> str:
    if prompt_config.use_adjacency_notation:
        return _format_knowledge_map_as_adjacency(
            edges,
            use_array_representation=prompt_config.use_array_representation,
        )

    return _format_knowledge_map_as_edge_list(
        edges,
        use_array_representation=prompt_config.use_array_representation,
    )


def _format_knowledge_map_as_adjacency(
    edges: list[Edge],
    *,
    use_array_representation: bool,
) -> str:
    ordered_nodes = _collect_ordered_nodes(edges)
    adjacency_matrix = _build_adjacency_matrix(
        edges,
        ordered_nodes=ordered_nodes,
    )

    if use_array_representation:
        return str(
            {
                "nodes": ordered_nodes,
                "adjacency_matrix": adjacency_matrix,
            }
        )

    node_tags = "".join(f"<node>{node}</node>" for node in ordered_nodes)
    return (
        "<knowledge-map>"
        f"<nodes>{node_tags}</nodes>"
        f"<adjacency-matrix>{adjacency_matrix}</adjacency-matrix>"
        "</knowledge-map>"
    )


def _format_knowledge_map_as_edge_list(
    edges: list[Edge],
    *,
    use_array_representation: bool,
) -> str:
    if use_array_representation:
        return str(edges)

    edge_tags: list[str] = []

    for source, target in edges:
        edge_tag = f"<edge source='{source}' target='{target}' />"
        edge_tags.append(edge_tag)

    joined_edge_tags = "".join(edge_tags)
    return f"<knowledge-map>{joined_edge_tags}</knowledge-map>"


def _collect_ordered_nodes(edges: list[Edge]) -> list[str]:
    ordered_nodes: list[str] = []
    seen_nodes: set[str] = set()

    for source, target in edges:
        for node in (source, target):
            if node in seen_nodes:
                continue
            ordered_nodes.append(node)
            seen_nodes.add(node)

    return ordered_nodes


def _build_adjacency_matrix(
    edges: list[Edge],
    *,
    ordered_nodes: list[str],
) -> list[list[int]]:
    node_positions: dict[str, int] = {}

    for index, node in enumerate(ordered_nodes):
        node_positions[node] = index

    matrix_size = len(ordered_nodes)
    adjacency_matrix = [[0 for _ in range(matrix_size)] for _ in range(matrix_size)]

    for source, target in edges:
        source_index = node_positions[source]
        target_index = node_positions[target]
        adjacency_matrix[source_index][target_index] = 1

    return adjacency_matrix


def extract_vote_list_from_chat_content(content: str) -> list[str]:
    parsed_content = json.loads(content)
    votes = parsed_content["votes"]
    normalized_votes = [str(vote) for vote in votes]
    return normalized_votes


def build_edge_generator(chat_client: ChatCompletionClient) -> "EdgeGenerator":
    def generate_edges(*, subgraph1: list[Edge], subgraph2: list[Edge]) -> list[Edge]:
        prompt = build_direct_edge_prompt(subgraph1=subgraph1, subgraph2=subgraph2)
        schema = {
            "type": "object",
            "properties": {
                "edges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "target": {"type": "string"},
                        },
                        "required": ["source", "target"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["edges"],
            "additionalProperties": False,
        }
        response = chat_client.complete_json(
            prompt=prompt,
            schema_name="edge_list",
            schema=schema,
        )
        raw_edges: list[dict[str, object]] = typing_cast(list[dict[str, object]], response["edges"])
        normalized_edges: list[Edge] = []

        for raw_edge in raw_edges:
            source = str(raw_edge["source"])
            target = str(raw_edge["target"])
            normalized_edge = (source, target)
            normalized_edges.append(normalized_edge)

        return normalized_edges

    return generate_edges


def build_cove_verifier(chat_client: ChatCompletionClient) -> "CoveVerifier":
    def verify_edges(candidate_edges: list[Edge]) -> list[Edge]:
        prompt = build_cove_prompt(candidate_edges)
        schema = {
            "type": "object",
            "properties": {
                "votes": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["votes"],
            "additionalProperties": False,
        }
        response = chat_client.complete_json(
            prompt=prompt,
            schema_name="vote_list",
            schema=schema,
        )
        votes: list[str] = typing_cast(list[str], response["votes"])
        normalized_votes = [str(vote) for vote in votes]
        verified_edges = apply_cove_verification(candidate_edges, normalized_votes)
        return verified_edges

    return verify_edges


class EdgeGenerator(Protocol):
    def __call__(self, *, subgraph1: list[Edge], subgraph2: list[Edge]) -> list[Edge]: ...


class CoveVerifier(Protocol):
    def __call__(self, candidate_edges: list[Edge]) -> list[Edge]: ...
