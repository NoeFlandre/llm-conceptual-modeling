import pytest

from llm_conceptual_modeling.common.model_catalog import (
    list_paper_model_catalog,
    resolve_model_alias,
)


def test_resolve_model_alias_maps_paper_chat_models_to_openrouter_ids() -> None:
    assert (
        resolve_model_alias(
            provider="openrouter",
            model="paper:deepseek-v3-0324",
            role="chat",
        )
        == "deepseek/deepseek-chat-v3-0324"
    )
    assert (
        resolve_model_alias(
            provider="openrouter",
            model="paper:gpt-5",
            role="chat",
        )
        == "openai/gpt-5"
    )


def test_resolve_model_alias_maps_paper_embedding_model_to_openrouter_id() -> None:
    assert (
        resolve_model_alias(
            provider="openrouter",
            model="paper:text-embedding-3-large",
            role="embedding",
        )
        == "text-embedding-3-large"
    )


def test_resolve_model_alias_rejects_provider_mismatch() -> None:
    with pytest.raises(ValueError, match="requires provider 'openrouter'"):
        resolve_model_alias(
            provider="mistral",
            model="paper:gpt-5",
            role="chat",
        )


def test_list_paper_model_catalog_includes_all_paper_chat_models() -> None:
    catalog = list_paper_model_catalog(role="chat")
    paper_labels = [entry["paper_label"] for entry in catalog]

    assert paper_labels == [
        "DeepSeek Chat v3.0324",
        "DeepSeek Chat v3.1",
        "Gemini 2.0 Flash",
        "Gemini 2.5 Pro",
        "GPT-4o",
        "GPT-5",
    ]


def test_resolve_model_alias_maps_hf_transformers_chat_models() -> None:
    assert (
        resolve_model_alias(
            provider="hf-transformers",
            model="paper:hf-ministral-3-8b-instruct-2512",
            role="chat",
        )
        == "mistralai/Ministral-3-8B-Instruct-2512"
    )
    assert (
        resolve_model_alias(
            provider="hf-transformers",
            model="paper:hf-qwen3.5-9b",
            role="chat",
        )
        == "Qwen/Qwen3.5-9B"
    )


def test_resolve_model_alias_maps_hf_transformers_embedding_model() -> None:
    assert (
        resolve_model_alias(
            provider="hf-transformers",
            model="paper:hf-qwen3-embedding-8b",
            role="embedding",
        )
        == "Qwen/Qwen3-Embedding-8B"
    )
