import json
from pathlib import Path

from llm_conceptual_modeling.common.io import read_json_dict, write_json_dict


def test_read_json_dict_returns_empty_dict_when_path_is_missing(tmp_path: Path) -> None:
    assert read_json_dict(tmp_path / "missing.json") == {}


def test_read_json_dict_loads_json_object_payload(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.json"
    payload_path.write_text('{"status":"finished","count":2}', encoding="utf-8")

    assert read_json_dict(payload_path) == {"status": "finished", "count": 2}


def test_write_json_dict_persists_sorted_indented_object_payload(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.json"

    write_json_dict(payload_path, {"b": 2, "a": 1})

    assert payload_path.read_text(encoding="utf-8") == '{\n  "a": 1,\n  "b": 2\n}'
    assert json.loads(payload_path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
