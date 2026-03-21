import json
from typing import Protocol
from urllib import request


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


def build_tree_expansion_prompt(
    *,
    source_labels: list[str],
    child_count: int,
) -> str:
    prompt = (
        "You are a helpful assistant who can creatively suggest relevant ideas. "
        "Your input is a set of concept names. "
        f"Your task is to recommend {child_count} related concept names "
        "for each of the names in the input. "
        "Do not suggest names that are already in the input. "
        "Return your proposed names in a dictionary format with source "
        "labels as keys and arrays of strings as values. "
        f"Input concepts: {source_labels}"
    )
    return prompt


class ChildProposer(Protocol):
    def __call__(
        self,
        source_labels: list[str],
        *,
        child_count: int,
    ) -> dict[str, list[str]]: ...


def build_child_proposer(chat_client: ChatCompletionClient) -> ChildProposer:
    def propose_children(
        source_labels: list[str],
        *,
        child_count: int,
    ) -> dict[str, list[str]]:
        prompt = build_tree_expansion_prompt(
            source_labels=source_labels,
            child_count=child_count,
        )
        schema = {
            "type": "object",
            "properties": {
                "children_by_label": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                }
            },
            "required": ["children_by_label"],
            "additionalProperties": False,
        }
        response = chat_client.complete_json(
            prompt=prompt,
            schema_name="children_by_label",
            schema=schema,
        )
        raw_children_by_label = response["children_by_label"]
        normalized_children_by_label: dict[str, list[str]] = {}

        for label, child_labels in raw_children_by_label.items():
            normalized_label = str(label)
            normalized_child_labels = [str(child_label) for child_label in child_labels]
            normalized_children_by_label[normalized_label] = normalized_child_labels

        return normalized_children_by_label

    return propose_children


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
