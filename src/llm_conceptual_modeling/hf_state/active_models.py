from __future__ import annotations

from pathlib import Path

import yaml

from llm_conceptual_modeling.hf_batch.utils import slugify_model


def resolve_active_chat_models(*roots: str | Path) -> set[str]:
    for root in roots:
        root_path = Path(root)
        for candidate in _runtime_config_candidates(root_path):
            if not candidate.exists():
                continue
            models = _load_chat_models(candidate)
            if models:
                return models
    return set()


def resolve_active_chat_model_slugs(*roots: str | Path) -> set[str]:
    return {slugify_model(model) for model in resolve_active_chat_models(*roots)}


def _load_chat_models(path: Path) -> set[str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return set()
    models = raw.get("models", {})
    if not isinstance(models, dict):
        return set()
    chat_models = models.get("chat_models", [])
    if not isinstance(chat_models, list):
        return set()
    return {str(model) for model in chat_models if str(model)}


def _runtime_config_candidates(root: Path) -> list[Path]:
    return [
        root / "runtime_config.yaml",
        root / "preview_resume" / "resolved_run_config.yaml",
        root / "preview" / "resolved_run_config.yaml",
    ]
