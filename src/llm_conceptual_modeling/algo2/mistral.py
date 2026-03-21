import json
from dataclasses import dataclass
from typing import Protocol
from urllib import request

Edge = tuple[str, str]


@dataclass(frozen=True)
class Method2PromptConfig:
    use_adjacency_notation: bool
    use_array_representation: bool
    include_explanation: bool
    include_example: bool
    include_counterexample: bool


class PostJsonFunction(Protocol):
    def __call__(
        self,
        *,
        url: str,
        api_key: str,
        payload: dict[str, object],
    ) -> dict[str, object]: ...


class MistralChatClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        post_json: PostJsonFunction | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._post_json = post_json or _post_json

    def complete_json(
        self,
        *,
        prompt: str,
        schema_name: str,
        schema: dict[str, object],
    ) -> dict[str, object]:
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema,
                },
            },
        }
        response = self._post_json(
            url="https://api.mistral.ai/v1/chat/completions",
            api_key=self._api_key,
            payload=payload,
        )
        content = response["choices"][0]["message"]["content"]
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


def build_label_expansion_prompt(
    seed_labels: list[str],
    *,
    subgraph1: list[Edge] | None = None,
    subgraph2: list[Edge] | None = None,
    prompt_config: Method2PromptConfig | None = None,
) -> str:
    resolved_prompt_config = prompt_config or Method2PromptConfig(
        use_adjacency_notation=False,
        use_array_representation=False,
        include_explanation=False,
        include_example=False,
        include_counterexample=False,
    )
    prompt_sections: list[str] = []
    prompt_sections.append("You are a helpful assistant who can creatively suggest relevant ideas.")

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

    if subgraph1 is not None and subgraph2 is not None:
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
    else:
        label_text = ", ".join(seed_labels)
        prompt_sections.append(f"Input concepts: {label_text}")

    prompt_sections.append(
        "Your task is to recommend 5 new related concept names for the input concepts. "
        "Do not suggest names that are already in the input. "
        "Return a JSON object with a top-level 'labels' array containing only strings."
    )
    prompt_sections.append(
        "Your output must only be the list of proposed concepts. "
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
        "We have the list of concepts ['capacity to hire', 'bad employees', "
        "'good reputation']. In this example, you could recommend these 9 new concepts: "
        "'employment potential', 'hiring capability', 'staffing ability', "
        "'underperformers', 'inefficient staff', 'problematic workers', "
        "'positive image', 'favorable standing', 'high regard'."
    )


def _build_counterexample_section() -> str:
    return (
        "Here is an example of a bad output that we do not want to see. "
        "A bad output would propose unrelated concepts such as 'moon', 'dog', "
        "or 'thermodynamics'."
    )


def _format_knowledge_map(
    edges: list[Edge],
    *,
    prompt_config: Method2PromptConfig,
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


def build_edge_suggestion_prompt(
    expanded_label_context: list[str],
    *,
    subgraph1: list[Edge] | None = None,
    subgraph2: list[Edge] | None = None,
    prompt_config: Method2PromptConfig | None = None,
) -> str:
    resolved_prompt_config = prompt_config or Method2PromptConfig(
        use_adjacency_notation=False,
        use_array_representation=False,
        include_explanation=False,
        include_example=False,
        include_counterexample=False,
    )
    prompt_sections: list[str] = []
    prompt_sections.append("You are a helpful assistant who can creatively suggest relevant ideas.")

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

    if subgraph1 is not None and subgraph2 is not None:
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

    label_text = ", ".join(expanded_label_context)
    prompt_sections.append(
        "Given a set of concept names, suggest edges that directly link concepts "
        "using the provided list. "
        "Use only exact concept names from the provided list. "
        "Return a JSON object with a top-level 'edges' array. "
        "Each edge must be an object with 'source' and 'target' string fields. "
        f"Available concepts: {label_text}"
    )
    prompt = " ".join(prompt_sections)
    return prompt


def extract_label_list_from_chat_content(content: str) -> list[str]:
    parsed_content = json.loads(content)
    labels = parsed_content["labels"]
    normalized_labels = [str(label) for label in labels]
    return normalized_labels


def build_label_proposer(chat_client: ChatCompletionClient) -> "LabelProposer":
    def propose_labels(current_labels: list[str]) -> list[str]:
        prompt = build_label_expansion_prompt(current_labels)
        schema = {
            "type": "object",
            "properties": {
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["labels"],
            "additionalProperties": False,
        }
        response = chat_client.complete_json(
            prompt=prompt,
            schema_name="label_list",
            schema=schema,
        )
        labels = response["labels"]
        normalized_labels = [str(label) for label in labels]
        return normalized_labels

    return propose_labels


def build_edge_suggester(chat_client: ChatCompletionClient) -> "EdgeSuggester":
    def suggest_edges(expanded_label_context: list[str]) -> list[Edge]:
        prompt = build_edge_suggestion_prompt(expanded_label_context)
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
        raw_edges = response["edges"]
        normalized_edges: list[Edge] = []

        for raw_edge in raw_edges:
            source = str(raw_edge["source"])
            target = str(raw_edge["target"])
            normalized_edge = (source, target)
            normalized_edges.append(normalized_edge)

        return normalized_edges

    return suggest_edges


class LabelProposer(Protocol):
    def __call__(self, current_labels: list[str]) -> list[str]: ...


class EdgeSuggester(Protocol):
    def __call__(self, expanded_label_context: list[str]) -> list[Edge]: ...


def _post_json(
    *,
    url: str,
    api_key: str,
    payload: dict[str, object],
) -> dict[str, object]:
    body_text = json.dumps(payload)
    body_bytes = body_text.encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    http_request = request.Request(
        url=url,
        data=body_bytes,
        headers=headers,
        method="POST",
    )
    with request.urlopen(http_request) as response:
        response_bytes = response.read()
    response_text = response_bytes.decode("utf-8")
    parsed_response = json.loads(response_text)
    return parsed_response
