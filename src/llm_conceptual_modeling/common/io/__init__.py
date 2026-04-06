from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def coerce_int(value: object, *, default: int = 0) -> int:
    try:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, str | bytes | bytearray):
            return int(value)
        index = getattr(value, "__index__", None)
        if callable(index):
            return int(index())
        integer = getattr(value, "__int__", None)
        if callable(integer):
            return int(integer())
    except (TypeError, ValueError):
        return default
    return default


def read_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Expected JSON object payload")
    return dict(payload)


def write_json_dict(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
