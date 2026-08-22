from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeGuard


def normalize_structured_response(
    parsed_content: object,
    *,
    schema_name: str,
) -> dict[str, object]:
    if isinstance(parsed_content, Mapping):
        return _normalize_mapping_response(
            parsed_content,
            schema_name=schema_name,
        )

    sequence_response = _normalize_sequence_response(parsed_content, schema_name)
    if sequence_response is not None:
        return sequence_response

    message = (
        "Unsupported structured response shape for schema "
        f"{schema_name}: {type(parsed_content).__name__}"
    )
    raise ValueError(message)


def _normalize_mapping_response(
    mapping: Mapping[Any, object],
    *,
    schema_name: str,
) -> dict[str, object]:
    if schema_name == "children_by_label":
        return {
            "children_by_label": _normalize_children_mapping_items(mapping)
        }
    if schema_name == "edge_list":
        return _normalize_list_mapping_field(
            mapping,
            "edges",
            _normalize_edge_item,
            schema_name=schema_name,
        )
    if schema_name == "vote_list":
        return _normalize_list_mapping_field(
            mapping,
            "votes",
            _normalize_vote_item,
            schema_name=schema_name,
        )
    if schema_name == "label_list":
        return _normalize_list_mapping_field(
            mapping,
            "labels",
            _normalize_label_item,
            schema_name=schema_name,
        )
    return dict(mapping)


def _normalize_list_mapping_field(
    mapping: Mapping[Any, object],
    field_name: str,
    normalizer: Callable[[object], object],
    *,
    schema_name: str,
) -> dict[str, object]:
    if field_name not in mapping:
        return dict(mapping)
    values = mapping[field_name]
    if not isinstance(values, list):
        raise ValueError(
            f"Structured {schema_name} response must contain a list of {field_name}"
        )
    return {
        field_name: [normalizer(item) for item in values],
    }


def _normalize_sequence_response(
    parsed_content: object,
    schema_name: str,
) -> dict[str, object] | None:
    if not _is_sequence_payload(parsed_content):
        return None
    items = parsed_content
    normalizer = _SEQUENCE_RESPONSE_NORMALIZERS.get(schema_name)
    return None if normalizer is None else normalizer(items)


def _normalize_edge_sequence_response(
    items: Sequence[object],
) -> dict[str, object]:
    return {"edges": _normalize_edge_list_items(items)}


def _normalize_vote_sequence_response(
    items: Sequence[object],
) -> dict[str, object]:
    return {"votes": [_normalize_vote_item(item) for item in items]}


def _normalize_label_sequence_response(
    items: Sequence[object],
) -> dict[str, object]:
    return {"labels": [_normalize_label_item(item) for item in items]}


def _normalize_children_sequence_response(
    items: Sequence[object],
) -> dict[str, object] | None:
    children_mapping = _normalize_children_sequence_payload(items)
    if children_mapping is None:
        return None
    return {"children_by_label": children_mapping}


_SEQUENCE_RESPONSE_NORMALIZERS: dict[
    str, Callable[[Sequence[object]], dict[str, object] | None]
] = {
    "edge_list": _normalize_edge_sequence_response,
    "vote_list": _normalize_vote_sequence_response,
    "label_list": _normalize_label_sequence_response,
    "children_by_label": _normalize_children_sequence_response,
}


def _normalize_edge_item(item: object) -> dict[str, str]:
    if isinstance(item, Mapping):
        item_mapping = item
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


def _normalize_vote_item(item: object) -> str:
    return _normalize_string_item(item, "vote")


def _normalize_label_item(item: object) -> str:
    return _normalize_string_item(item, "label")


def _normalize_children_mapping_items(
    mapping: Mapping[Any, object],
) -> dict[str, list[str]]:
    raw_children = mapping.get("children_by_label")
    if isinstance(raw_children, Mapping):
        mapping = raw_children

    normalized: dict[str, list[str]] = {}
    for parent_label, child_value in mapping.items():
        parent_text = _normalize_string_item(parent_label, "parent label")
        normalized[parent_text] = _normalize_children_value(child_value)
    return normalized


def _normalize_children_sequence_payload(
    items: Sequence[object],
) -> dict[str, list[str]] | None:
    if _is_single_children_sequence_payload(items):
        return _normalize_children_pair(items)

    mapping: dict[str, list[str]] = {}
    for item in items:
        if not _is_children_pair(item):
            return None
        pair = item
        parent_text = _normalize_string_item(pair[0], "parent label")
        mapping[parent_text] = _normalize_children_value(pair[1])
    return mapping


def _is_single_children_sequence_payload(items: Sequence[object]) -> bool:
    return len(items) == 2 and not isinstance(items[0], Mapping | list | tuple)


def _normalize_children_pair(items: Sequence[object]) -> dict[str, list[str]]:
    parent_text = _normalize_string_item(items[0], "parent label")
    return {parent_text: _normalize_children_value(items[1])}


def _is_children_pair(item: object) -> TypeGuard[Sequence[object]]:
    return isinstance(item, list | tuple) and len(item) >= 2


def _normalize_children_value(child_value: object) -> list[str]:
    if _is_children_sequence(child_value):
        return [
            str(item).strip()
            for item in child_value
            if _can_normalize_child_label(item)
        ]
    if child_value is None:
        return []
    return [_normalize_string_item(child_value, "child label")]


def _is_children_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    )


def _normalize_edge_list_items(items: Sequence[object]) -> list[dict[str, str]]:
    if _is_flat_scalar_edge_list(items):
        flat_items = _normalize_flat_edge_endpoints(items)
        return _pair_flat_edge_endpoints(flat_items)

    return [_normalize_edge_item(item) for item in items]


def _is_flat_scalar_edge_list(items: Sequence[object]) -> bool:
    return bool(items) and all(_is_scalar_edge_endpoint(item) for item in items)


def _normalize_flat_edge_endpoints(items: Sequence[object]) -> list[str]:
    flat_items = [_normalize_string_item(item, "edge endpoint") for item in items]
    if len(flat_items) % 2 == 0:
        return flat_items
    recovered_items = _drop_dangling_noisy_edge_endpoint(flat_items)
    if recovered_items is not None:
        return recovered_items
    if _should_drop_short_odd_flat_edge_list(flat_items):
        return []
    raise ValueError(
        "Structured edge_list flat string response must contain an even number of items"
    )


def _pair_flat_edge_endpoints(items: list[str]) -> list[dict[str, str]]:
    return [
        {
            "source": items[index],
            "target": items[index + 1],
        }
        for index in range(0, len(items), 2)
    ]


def _is_scalar_edge_endpoint(item: object) -> bool:
    if item is None or isinstance(item, bool):
        return False
    return not isinstance(item, Mapping | list | tuple)


def _drop_dangling_noisy_edge_endpoint(items: list[str]) -> list[str] | None:
    if len(items) < 5:
        return None
    if len(items) % 2 == 0:
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
    if any(character in item for character in ("[", "]", "{", "}")):
        return True
    return False


def _should_drop_short_odd_flat_edge_list(items: list[str]) -> bool:
    if len(items) != 7:
        return False
    return all(any(character.isalnum() for character in item) for item in items)


def _is_sequence_payload(item: object) -> TypeGuard[Sequence[object]]:
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
