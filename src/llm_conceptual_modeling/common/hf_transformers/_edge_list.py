from __future__ import annotations

import re


def _recover_bare_comma_separated_edge_pair(text: str) -> list[tuple[str, str]] | None:
    if any(token in text for token in "[](){}:\n"):
        return None
    parts = [part.strip().strip("'\"") for part in text.split(",", 1)]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return [(parts[0], parts[1])]


def _recover_bracketed_edge_pairs(text: str) -> list[tuple[str, str]] | None:
    bracket_matches = re.findall(r"\[([^\[\](){}]+)\]", text)
    if not bracket_matches:
        return None
    parsed_edges: list[tuple[str, str]] = []
    for bracket_text in bracket_matches:
        pair = _extract_edge_pair_from_bracket(bracket_text)
        if pair is None:
            continue
        parsed_edges.append(pair)
    return parsed_edges or None


def _extract_edge_pair_from_bracket(bracket_text: str) -> tuple[str, str] | None:
    quoted_items = re.findall(r"""['"]([^'"]+)['"]""", bracket_text)
    if len(quoted_items) == 2:
        return (quoted_items[0].strip(), quoted_items[1].strip())
    if "'" in bracket_text or '"' in bracket_text:
        return None
    parts = [part.strip() for part in bracket_text.split(",", 1)]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return (parts[0], parts[1])


def _extract_recoverable_edge_endpoints(text: str) -> list[str] | None:
    quoted_items = re.findall(r"""['"]([^'"]+)['"]""", text)
    if not quoted_items:
        return None
    normalized_items = [item.strip() for item in quoted_items]
    if len(normalized_items) % 2 != 0:
        return None
    if not all(_looks_like_edge_endpoint(item) for item in normalized_items):
        return None
    return normalized_items


def _looks_like_truncated_single_edge_endpoint(text: str) -> bool:
    stripped = text.strip()
    if not stripped or not any(token in stripped for token in ("[", "(")):
        return False
    quoted_items = re.findall(r"""['"]([^'"\n]+)""", stripped)
    if len(quoted_items) != 1:
        return False
    endpoint = quoted_items[0].strip()
    if not _looks_like_edge_endpoint(endpoint):
        return False
    lowered = stripped.lower()
    return (
        "<think>" in lowered or "</think>" in lowered or not stripped.rstrip().endswith(("]", ")"))
    )


def _looks_like_edge_endpoint(text: str) -> bool:
    if not text:
        return False
    return bool(re.search(r"[A-Za-z0-9]", text))
