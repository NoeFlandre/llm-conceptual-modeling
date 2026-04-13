from pathlib import Path

import pytest

from llm_conceptual_modeling.hf_state.active_models import (
    resolve_active_chat_models,
    resolve_active_chat_model_slugs,
)


def test_hf_state_active_models_package_imports_and_resolves_models(tmp_path: Path) -> None:
    root = tmp_path
    runtime_config = root / "runtime_config.yaml"
    runtime_config.write_text(
        "\n".join(
            [
                "models:",
                "  chat_models:",
                "    - Qwen/Qwen3.5-9B",
                "    - mistralai/Ministral-3-8B-Instruct-2512",
            ]
        ),
        encoding="utf-8",
    )

    assert resolve_active_chat_models(root) == {
        "Qwen/Qwen3.5-9B",
        "mistralai/Ministral-3-8B-Instruct-2512",
    }
    assert resolve_active_chat_model_slugs(root) == {
        "Qwen__Qwen3.5-9B",
        "mistralai__Ministral-3-8B-Instruct-2512",
    }
