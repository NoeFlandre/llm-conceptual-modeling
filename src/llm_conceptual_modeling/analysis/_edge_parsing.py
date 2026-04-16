"""Edge-list parsing helpers for baseline comparison.

Provides two parsing namespaces:
- `_parse_edges` / `_parse_edges_cached`: Python-literal parsing for algo1/2 outputs
- `_parse_algo3_edge_list` / `_parse_algo3_edge_list_cached`: AST+regex fallback for algo3 outputs
"""

from __future__ import annotations

import ast
import re
from functools import lru_cache

from llm_conceptual_modeling.common.literals import parse_python_literal


@lru_cache(maxsize=None)
def _parse_edges_cached(text: str) -> tuple[tuple[str, str], ...]:
    if text == "None":
        return tuple()
    try:
        parsed = parse_python_literal(text)
        if isinstance(parsed, (list, set, tuple)):
            edges: list[tuple[str, str]] = []
            for item in parsed:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    edges.append((str(item[0]).strip(), str(item[1]).strip()))
            return tuple(edges)
    except (ValueError, SyntaxError):
        return tuple()
    return tuple()


def _parse_edges(value: str | object) -> list[tuple[str, str]]:
    return list(_parse_edges_cached(str(value)))


@lru_cache(maxsize=None)
def _parse_algo3_edge_list_cached(text: str) -> tuple[tuple[str, str], ...]:
    if text == "None":
        return tuple()
    text = text.strip()
    if not text or text.lower() in {"empty", "nan"}:
        return tuple()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, set, tuple)):
            parsed_edges: list[tuple[str, str]] = []
            for item in parsed:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    parsed_edges.append((str(item[0]).strip(), str(item[1]).strip()))
            if parsed_edges:
                return tuple(parsed_edges)
    except (ValueError, SyntaxError):
        pass

    pairs = re.findall(r"\(([^()]+?,[^()]+?)\)", text)
    edges: list[tuple[str, str]] = []
    for pair in pairs:
        parts = pair.split(",", 1)
        if len(parts) != 2:
            continue
        left = parts[0].strip().strip("'\"")
        right = parts[1].strip().strip("'\"")
        if left and right:
            edges.append((left, right))
    return tuple(edges)


def _parse_algo3_edge_list(value: str | object) -> list[tuple[str, str]]:
    return list(_parse_algo3_edge_list_cached(str(value)))
