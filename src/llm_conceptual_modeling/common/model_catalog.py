from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, TypeAlias

ModelRole: TypeAlias = Literal["chat", "embedding"]


@dataclass(frozen=True)
class ModelCatalogEntry:
    alias: str
    provider: str
    model: str
    paper_label: str
    role: ModelRole

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


PAPER_CHAT_MODELS: tuple[ModelCatalogEntry, ...] = (
    ModelCatalogEntry(
        alias="paper:deepseek-v3-0324",
        provider="openrouter",
        model="deepseek/deepseek-chat-v3-0324",
        paper_label="DeepSeek Chat v3.0324",
        role="chat",
    ),
    ModelCatalogEntry(
        alias="paper:deepseek-v3.1",
        provider="openrouter",
        model="deepseek/deepseek-chat-v3.1",
        paper_label="DeepSeek Chat v3.1",
        role="chat",
    ),
    ModelCatalogEntry(
        alias="paper:gemini-2.0-flash",
        provider="openrouter",
        model="google/gemini-2.0-flash-001",
        paper_label="Gemini 2.0 Flash",
        role="chat",
    ),
    ModelCatalogEntry(
        alias="paper:gemini-2.5-pro",
        provider="openrouter",
        model="google/gemini-2.5-pro",
        paper_label="Gemini 2.5 Pro",
        role="chat",
    ),
    ModelCatalogEntry(
        alias="paper:gpt-4o",
        provider="openrouter",
        model="openai/gpt-4o-2024-05-13",
        paper_label="GPT-4o",
        role="chat",
    ),
    ModelCatalogEntry(
        alias="paper:gpt-5",
        provider="openrouter",
        model="openai/gpt-5",
        paper_label="GPT-5",
        role="chat",
    ),
)

HF_CHAT_MODELS: tuple[ModelCatalogEntry, ...] = (
    ModelCatalogEntry(
        alias="paper:hf-ministral-3-8b-instruct-2512",
        provider="hf-transformers",
        model="mistralai/Ministral-3-8B-Instruct-2512",
        paper_label="Ministral-3-8B-Instruct-2512",
        role="chat",
    ),
    ModelCatalogEntry(
        alias="paper:hf-qwen3.5-9b",
        provider="hf-transformers",
        model="Qwen/Qwen3.5-9B",
        paper_label="Qwen3.5-9B",
        role="chat",
    ),
    ModelCatalogEntry(
        alias="paper:hf-olmo-3-7b-instruct",
        provider="hf-transformers",
        model="allenai/Olmo-3-7B-Instruct",
        paper_label="OLMo-3-7B-Instruct",
        role="chat",
    ),
)

PAPER_EMBEDDING_MODELS: tuple[ModelCatalogEntry, ...] = (
    ModelCatalogEntry(
        alias="paper:text-embedding-3-large",
        provider="openrouter",
        model="text-embedding-3-large",
        paper_label="text-embedding-3-large",
        role="embedding",
    ),
)

HF_EMBEDDING_MODELS: tuple[ModelCatalogEntry, ...] = (
    ModelCatalogEntry(
        alias="paper:hf-qwen3-embedding-8b",
        provider="hf-transformers",
        model="Qwen/Qwen3-Embedding-8B",
        paper_label="Qwen3-Embedding-8B",
        role="embedding",
    ),
)

_ALIASES_BY_ROLE: dict[ModelRole, dict[str, ModelCatalogEntry]] = {
    "chat": {entry.alias: entry for entry in PAPER_CHAT_MODELS + HF_CHAT_MODELS},
    "embedding": {entry.alias: entry for entry in PAPER_EMBEDDING_MODELS + HF_EMBEDDING_MODELS},
}


def list_paper_model_catalog(role: ModelRole | None = None) -> list[dict[str, str]]:
    entries: tuple[ModelCatalogEntry, ...]
    if role is None:
        entries = PAPER_CHAT_MODELS + PAPER_EMBEDDING_MODELS
    else:
        entries = PAPER_CHAT_MODELS if role == "chat" else PAPER_EMBEDDING_MODELS
    return [entry.to_dict() for entry in entries]


def resolve_model_alias(*, provider: str, model: str, role: ModelRole) -> str:
    entry = _ALIASES_BY_ROLE[role].get(model)
    if entry is None:
        return model
    if provider != entry.provider:
        raise ValueError(
            f"Model alias {model!r} requires provider {entry.provider!r}, got {provider!r}"
        )
    return entry.model
