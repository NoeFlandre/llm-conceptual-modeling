from __future__ import annotations

from collections.abc import Mapping
from typing import cast


def normalize_structured_response(
    parsed_content: object,
    *,
    schema_name: str,
) -> dict[str, object]:
    if isinstance(parsed_content, Mapping):
        parsed_mapping = cast(Mapping[str, object], parsed_content)
        if schema_name == "edge_list" and "edges" in parsed_mapping:
            edges = parsed_mapping["edges"]
            if not isinstance(edges, list):
                raise ValueError("Structured edge_list response must contain a list of edges")
            return {"edges": [_normalize_edge_item(item) for item in edges]}
        if schema_name == "vote_list" and "votes" in parsed_mapping:
            votes = parsed_mapping["votes"]
            if not isinstance(votes, list):
                raise ValueError("Structured vote_list response must contain a list of votes")
            return {"votes": [_normalize_string_item(item, "vote") for item in votes]}
        if schema_name == "label_list" and "labels" in parsed_mapping:
            labels = parsed_mapping["labels"]
            if not isinstance(labels, list):
                raise ValueError("Structured label_list response must contain a list of labels")
            return {"labels": [_normalize_string_item(item, "label") for item in labels]}
        return dict(parsed_mapping)

    if schema_name == "edge_list" and isinstance(parsed_content, list):
        return {"edges": [_normalize_edge_item(item) for item in parsed_content]}

    if schema_name == "vote_list" and isinstance(parsed_content, list):
        return {"votes": [_normalize_string_item(item, "vote") for item in parsed_content]}

    if schema_name == "label_list" and isinstance(parsed_content, list):
        return {"labels": [_normalize_string_item(item, "label") for item in parsed_content]}

    message = (
        "Unsupported structured response shape for schema "
        f"{schema_name}: {type(parsed_content).__name__}"
    )
    raise ValueError(message)


def _normalize_edge_item(item: object) -> dict[str, str]:
    if isinstance(item, Mapping):
        item_mapping = cast(Mapping[str, object], item)
        source = item_mapping.get("source")
        target = item_mapping.get("target")
        return {
            "source": _normalize_string_item(source, "edge source"),
            "target": _normalize_string_item(target, "edge target"),
        }

    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return {
            "source": _normalize_string_item(item[0], "edge source"),
            "target": _normalize_string_item(item[1], "edge target"),
        }

    raise ValueError(f"Invalid edge item shape: {item!r}")


def _normalize_string_item(item: object, item_name: str) -> str:
    if item is None:
        raise ValueError(f"Structured response returned a null {item_name}")
    text = str(item).strip()
    if not text or text.lower() == "none":
        raise ValueError(f"Structured response returned an empty {item_name}")
    return text
