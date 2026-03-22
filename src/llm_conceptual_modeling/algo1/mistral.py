from dataclasses import dataclass
from typing import Callable
from typing import cast as typing_cast

from mistralai import Mistral
from mistralai.models import ChatMessage

from llm_conceptual_modeling.common.types import ChatCompletionClient

from .cove import apply_cove_verification

Edge = tuple[str, str]


@dataclass
class MistralGenerationResult:
    raw_response: str
    edges: list[Edge]
    verified_edges: list[Edge]
    prompt_token_count: int | None = None
    completion_token_count: int | None = None


@dataclass
class MistralCoVeResult:
    raw_response: str
    votes: list[str]
    verified_edges: list[Edge]


class MistralChatClient:
    def __init__(self, api_key: str, model: str, sdk_client: "Mistral | None" = None):
        self.api_key = api_key
        self.model = model
        self._client = sdk_client if sdk_client is not None else Mistral(api_key=api_key)

    def complete_json(
        self,
        *,
        prompt: str,
        schema_name: str,
        schema: dict[str, object],
    ) -> dict[str, object]:
        import mistralai.exceptions

        response = self._client.chat.complete(
            model=self.model,
            messages=[ChatMessage(role="user", content=prompt)],
            response_format={"type": "json_object", "schema": schema},
        )
        try:
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Empty response from Mistral")
            import json

            return json.loads(content)
        except (ValueError, KeyError, mistralai.exceptions.MistralAPIError) as exc:
            raise RuntimeError(f" Mistral response parsing failed: {exc}") from exc


def build_edge_generation_prompt(
    subgraph1: list[Edge],
    subgraph2: list[Edge],
    prompt_config: object,
) -> str:
    subgraph1_str = "\n".join(f"  ({s}, {t})" for s, t in subgraph1)
    subgraph2_str = "\n".join(f"  ({s}, {t})" for s, t in subgraph2)
    return f"""\
{system_prompt}

## Task
Your task is to recommend more links between the two maps.

## Map A
{subgraph1_str}

## Map B
{subgraph2_str}

{user_prompt(prompt_config)}"""


def build_cove_prompt(candidate_edges: list[Edge]) -> str:
    candidate_str = "\n".join(f"  ({s}, {t})" for s, t in candidate_edges)
    return f"""\
{system_prompt}

## Task
For each candidate edge below, judge whether a causal relationship exists
from the source concept to the target concept. Return a YES if you believe
the relationship is credible. Return NO otherwise.

## Candidate Edges
{candidate_str}

{user_prompt_cot(prompt_config=None)}


You must respond with a JSON object with a "votes" key mapping each candidate edge to YES or NO.


{user_prompt(prompt_config=None)}"""


system_prompt = "You are a helpful assistant that is good at causal reasoning."


def user_prompt(prompt_config: object) -> str:
    return "Your task is to recommend more links between the two maps."


def user_prompt_cot(prompt_config: object) -> str:
    return ""


class CoveVerifier:
    def __init__(self, chat_client: ChatCompletionClient) -> None:
        self._chat_client = chat_client

    def verify_edges(self, candidate_edges: list[Edge]) -> list[Edge]:
        """Given a list of candidate edges, verify each using CoT + tool calling."""
        prompt = build_cove_prompt(candidate_edges)
        schema = {
            "type": "object",
            "properties": {
                "votes": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["YES", "NO"]},
                }
            },
            "required": ["votes"],
        }
        schema_name = "vote_list"
        response = self._chat_client.complete_json(
            prompt=prompt,
            schema_name=schema_name,
            schema=schema,
        )
        votes = typing_cast(list[str], response["votes"])
        verified_edges = apply_cove_verification(candidate_edges, votes)
        return verified_edges


def build_cove_verifier(chat_client: ChatCompletionClient) -> Callable[[list[Edge]], list[Edge]]:
    def verify_edges(candidate_edges: list[Edge]) -> list[Edge]:
        prompt = build_cove_prompt(candidate_edges)
        schema = {
            "type": "object",
            "properties": {
                "votes": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["YES", "NO"]},
                }
            },
            "required": ["votes"],
        }
        schema_name = "vote_list"
        response = chat_client.complete_json(
            prompt=prompt,
            schema_name=schema_name,
            schema=schema,
        )
        votes: list[str] = typing_cast(list[str], response["votes"])
        verified_edges = apply_cove_verification(candidate_edges, votes)
        return verified_edges

    return verify_edges


def generate_edges(
    chat_client: ChatCompletionClient,
    subgraph1: list[Edge],
    subgraph2: list[Edge],
    prompt_config: object,
) -> list[Edge]:
    """Prompt the LLM to generate candidate edges between two subgraphs."""
    prompt = build_edge_generation_prompt(subgraph1, subgraph2, prompt_config)
    schema = {
        "type": "object",
        "properties": {
            "edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"source": {"type": "string"}, "target": {"type": "string"}},
                    "required": ["source", "target"],
                },
            }
        },
        "required": ["edges"],
    }
    schema_name = "edge_list"
    response = chat_client.complete_json(
        prompt=prompt,
        schema_name=schema_name,
        schema=schema,
    )
    raw_edges: list[dict[str, object]] = typing_cast(
        list[dict[str, object]], response["edges"]
    )
    normalized_edges: list[Edge] = []
    for raw_edge in raw_edges:
        source = str(raw_edge["source"])
        target = str(raw_edge["target"])
        normalized_edges.append((source, target))
    return normalized_edges
