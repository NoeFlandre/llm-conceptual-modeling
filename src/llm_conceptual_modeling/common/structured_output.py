from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast


def normalize_structured_response(
    parsed_content: object,
    *,
    schema_name: str,
) -> dict[str, object]:
    if isinstance(parsed_content, Mapping):
        parsed_mapping = cast(Mapping[str, object], parsed_content)
        if schema_name == "children_by_label":
            return {
                "children_by_label": _normalize_children_mapping_items(parsed_mapping)
            }
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

    if schema_name == "edge_list" and _is_sequence_payload(parsed_content):
        return {"edges": _normalize_edge_list_items(cast(Sequence[object], parsed_content))}

    if schema_name == "vote_list" and _is_sequence_payload(parsed_content):
        items = cast(Sequence[object], parsed_content)
        return {"votes": [_normalize_string_item(item, "vote") for item in items]}

    if schema_name == "label_list" and _is_sequence_payload(parsed_content):
        items = cast(Sequence[object], parsed_content)
        return {"labels": [_normalize_string_item(item, "label") for item in items]}

    if schema_name == "children_by_label" and _is_sequence_payload(parsed_content):
        children_mapping = _normalize_children_sequence_payload(
            cast(Sequence[object], parsed_content)
        )
        if children_mapping is not None:
            return {"children_by_label": children_mapping}

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


def _normalize_children_mapping_items(
    mapping: Mapping[str, object],
) -> dict[str, list[str]]:
    raw_children = mapping.get("children_by_label")
    if isinstance(raw_children, Mapping):
        mapping = cast(Mapping[str, object], raw_children)

    normalized: dict[str, list[str]] = {}
    for parent_label, child_value in mapping.items():
        parent_text = _normalize_string_item(parent_label, "parent label")
        if isinstance(child_value, Sequence) and not isinstance(
            child_value, str | bytes | bytearray
        ):
            child_labels = [
                _normalize_string_item(item, "child label")
                for item in cast(Sequence[object], child_value)
                if _can_normalize_child_label(item)
            ]
        elif child_value is None:
            child_labels = []
        else:
            child_labels = [_normalize_string_item(child_value, "child label")]
        normalized[parent_text] = child_labels
    return normalized


def _normalize_children_sequence_payload(
    items: Sequence[object],
) -> dict[str, list[str]] | None:
    if len(items) == 2 and not isinstance(items[0], Mapping | list | tuple):
        parent_text = _normalize_string_item(items[0], "parent label")
        child_value = items[1]
        if isinstance(child_value, Sequence) and not isinstance(
            child_value, str | bytes | bytearray
        ):
            child_labels = [
                _normalize_string_item(item, "child label")
                for item in cast(Sequence[object], child_value)
                if _can_normalize_child_label(item)
            ]
        elif child_value is None:
            child_labels = []
        else:
            child_labels = [_normalize_string_item(child_value, "child label")]
        return {parent_text: child_labels}

    mapping: dict[str, list[str]] = {}
    for item in items:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            return None
        parent_text = _normalize_string_item(item[0], "parent label")
        child_value = item[1]
        if isinstance(child_value, Sequence) and not isinstance(
            child_value, str | bytes | bytearray
        ):
            child_labels = [
                _normalize_string_item(value, "child label")
                for value in cast(Sequence[object], child_value)
                if _can_normalize_child_label(value)
            ]
        elif child_value is None:
            child_labels = []
        else:
            child_labels = [_normalize_string_item(child_value, "child label")]
        mapping[parent_text] = child_labels
    return mapping


def _normalize_edge_list_items(items: Sequence[object]) -> list[dict[str, str]]:
    if items and all(_is_scalar_edge_endpoint(item) for item in items):
        flat_items = [_normalize_string_item(item, "edge endpoint") for item in items]
        if len(flat_items) % 2 != 0:
            if _should_drop_short_odd_flat_edge_list(flat_items):
                return []
            recovered_items = _drop_dangling_noisy_edge_endpoint(flat_items)
            if recovered_items is None:
                raise ValueError(
                    "Structured edge_list flat string response must "
                    "contain an even number of items"
                )
            flat_items = recovered_items
        paired_edges: list[dict[str, str]] = []
        for index in range(0, len(flat_items), 2):
            paired_edges.append(
                {
                    "source": flat_items[index],
                    "target": flat_items[index + 1],
                }
            )
        return paired_edges

    return [_normalize_edge_item(item) for item in items]


def _is_scalar_edge_endpoint(item: object) -> bool:
    if item is None or isinstance(item, bool):
        return False
    return not isinstance(item, Mapping | list | tuple)


def _drop_dangling_noisy_edge_endpoint(items: list[str]) -> list[str] | None:
    if len(items) < 5 or len(items) % 2 == 0:
        return None
    dangling_item = items[-1].strip()
    if not _looks_like_noisy_dangling_edge_endpoint(dangling_item):
        return None
    return items[:-1]


def _looks_like_noisy_dangling_edge_endpoint(item: str) -> bool:
    if not item:
        return False
    if item.count(",") >= 2:
        return True
    if any(character in item for character in "[]{}"):
        return True
    return False


def _should_drop_short_odd_flat_edge_list(items: list[str]) -> bool:
    if len(items) > 7 or len(items) % 2 == 0:
        return False
    return all(any(character.isalnum() for character in item) for item in items)


def _is_sequence_payload(item: object) -> bool:
    if isinstance(item, str | bytes | bytearray):
        return False
    return isinstance(item, Sequence)


def _normalize_string_item(item: object, item_name: str) -> str:
    if item is None:
        raise ValueError(f"Structured response returned a null {item_name}")
    text = str(item).strip()
    if not text or text.lower() == "none":
        raise ValueError(f"Structured response returned an empty {item_name}")
    return text


def _can_normalize_child_label(item: object) -> bool:
    if item is None:
        return False
    text = str(item).strip()
    return bool(text) and text.lower() != "none"
