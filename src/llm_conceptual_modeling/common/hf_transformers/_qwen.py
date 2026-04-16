"""Qwen-specific contrastive-search compatibility helpers.

Patches the contrastive-search custom generate to handle Qwen's dynamic cache
type, and provides a context manager to temporarily disable Qwen's stateful guard
during generation.
"""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from typing import Any

from llm_conceptual_modeling.common.hf_transformers._policy import (
    _QWEN_CHAT_MODEL,
    DecodingConfig,
)


def _get_load_qwen_dynamic_cache_type() -> Any:
    """Look up _load_qwen_dynamic_cache_type from the hf_transformers namespace."""
    import sys as _sys

    hf = _sys.modules.get("llm_conceptual_modeling.common.hf_transformers")
    if hf is not None:
        ref = getattr(hf, "_load_qwen_dynamic_cache_type", None)
        if ref is not None:
            return ref
    return _load_qwen_dynamic_cache_type


def _custom_generate_overrides(
    *,
    model: str,
    model_object: Any,
    decoding_config: DecodingConfig,
) -> dict[str, object]:
    if model != _QWEN_CHAT_MODEL or decoding_config.algorithm != "contrastive":
        return {}
    load_custom_generate = getattr(model_object, "load_custom_generate", None)
    if not callable(load_custom_generate):
        return {}
    custom_generate = load_custom_generate(
        "transformers-community/contrastive-search",
        trust_remote_code=True,
    )
    return {"custom_generate": _patch_qwen_contrastive_custom_generate(custom_generate)}


def _patch_qwen_contrastive_custom_generate(custom_generate: Any) -> Any:
    if not callable(custom_generate):
        return custom_generate
    qwen_dynamic_cache_type = _get_load_qwen_dynamic_cache_type()()
    if qwen_dynamic_cache_type is None:
        return custom_generate
    _ensure_qwen_dynamic_cache_compatibility(qwen_dynamic_cache_type)
    custom_generate_globals = getattr(custom_generate, "__globals__", None)
    if not isinstance(custom_generate_globals, dict):
        return custom_generate
    dynamic_cache_type = custom_generate_globals.get("DynamicCache")
    if dynamic_cache_type is None:
        return custom_generate
    if isinstance(dynamic_cache_type, tuple):
        if qwen_dynamic_cache_type in dynamic_cache_type:
            return custom_generate
        custom_generate_globals["DynamicCache"] = (*dynamic_cache_type, qwen_dynamic_cache_type)
        return custom_generate
    if dynamic_cache_type is qwen_dynamic_cache_type:
        return custom_generate
    custom_generate_globals["DynamicCache"] = (dynamic_cache_type, qwen_dynamic_cache_type)
    return custom_generate


def _load_qwen_dynamic_cache_type() -> type[Any] | None:
    try:
        module = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    except ImportError:
        return None
    dynamic_cache_type = getattr(module, "Qwen3_5DynamicCache", None)
    if isinstance(dynamic_cache_type, type):
        return dynamic_cache_type
    return None


def _ensure_qwen_dynamic_cache_compatibility(qwen_dynamic_cache_type: type[Any]) -> None:
    if not hasattr(qwen_dynamic_cache_type, "batch_repeat_interleave"):
        qwen_dynamic_cache_type.batch_repeat_interleave = _qwen_cache_batch_repeat_interleave
    if not hasattr(qwen_dynamic_cache_type, "crop"):
        qwen_dynamic_cache_type.crop = _qwen_cache_crop
    if not hasattr(qwen_dynamic_cache_type, "batch_select_indices"):
        qwen_dynamic_cache_type.batch_select_indices = _qwen_cache_batch_select_indices


def _qwen_cache_batch_repeat_interleave(self: Any, repeats: int) -> None:
    _mutate_qwen_cache_state_lists(
        self,
        lambda tensor: tensor.repeat_interleave(repeats, dim=0),
    )


def _qwen_cache_crop(self: Any, max_length: int) -> None:
    if max_length < 0:
        max_length = self.get_seq_length() - abs(max_length)
    if self.get_seq_length() <= max_length:
        return
    key_cache = getattr(self, "key_cache", None)
    value_cache = getattr(self, "value_cache", None)
    if isinstance(key_cache, list):
        for index, tensor in enumerate(key_cache):
            if tensor is not None:
                key_cache[index] = tensor[..., :max_length, :]
    if isinstance(value_cache, list):
        for index, tensor in enumerate(value_cache):
            if tensor is not None:
                value_cache[index] = tensor[..., :max_length, :]


def _qwen_cache_batch_select_indices(self: Any, indices: Any) -> None:
    _mutate_qwen_cache_state_lists(
        self,
        lambda tensor: tensor.index_select(0, indices.to(tensor.device)),
    )


def _mutate_qwen_cache_state_lists(self: Any, transform: Any) -> None:
    for attribute_name in ("key_cache", "value_cache", "conv_states", "recurrent_states"):
        cache_states = getattr(self, attribute_name, None)
        if not isinstance(cache_states, list):
            continue
        for index, tensor in enumerate(cache_states):
            if tensor is None:
                continue
            cache_states[index] = transform(tensor)


@contextmanager
def _temporarily_disable_stateful_guard(
    *,
    model: str,
    model_object: Any,
    decoding_config: DecodingConfig,
):
    if model != _QWEN_CHAT_MODEL or decoding_config.algorithm != "contrastive":
        yield
        return

    original_stateful = getattr(model_object, "_is_stateful", None)
    if original_stateful is not True:
        yield
        return

    model_object._is_stateful = False
    try:
        yield
    finally:
        model_object._is_stateful = original_stateful
