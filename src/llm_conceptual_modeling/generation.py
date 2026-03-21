import json


def build_generation_stub_payload(
    algorithm: str,
    *,
    fixture_only: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "algorithm": algorithm,
        "mode": "stub",
        "implemented": False,
        "requires_live_llm": True,
        "fixture_only": fixture_only,
    }
    if fixture_only:
        payload["next_step"] = "provide_fixture_dataset"
    else:
        payload["next_step"] = "implement_provider_adapter"
    return payload


def emit_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload))
