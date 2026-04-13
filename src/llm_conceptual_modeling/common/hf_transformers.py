from __future__ import annotations

import ast
import gc
import importlib.util
import importlib
import json
import random
import re
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

import numpy as np

from llm_conceptual_modeling.common.structured_output import normalize_structured_response

_QWEN_CHAT_MODEL = "Qwen/Qwen3.5-9B"
_MINISTRAL_CHAT_MODEL = "mistralai/Ministral-3-8B-Instruct-2512"
_SUPPORTED_CHAT_MODELS = {
    _MINISTRAL_CHAT_MODEL,
    _QWEN_CHAT_MODEL,
    "allenai/Olmo-3-7B-Instruct",
}
_SUPPORTED_EMBEDDING_MODELS = {
    "Qwen/Qwen3-Embedding-8B",
    "Qwen/Qwen3-Embedding-0.6B",
}
_TRUSTED_REMOTE_CODE_CHAT_MODELS = set(_SUPPORTED_CHAT_MODELS)


def supports_explicit_thinking_disable(model: str) -> bool:
    return model == _QWEN_CHAT_MODEL


def supports_decoding_config(*, model: str, decoding_config: "DecodingConfig") -> bool:
    return True


def runtime_generation_overrides(
    *,
    model: str,
    decoding_config: "DecodingConfig",
) -> dict[str, object]:
    if model == "allenai/Olmo-3-7B-Instruct" and decoding_config.algorithm == "contrastive":
        return {"low_memory": True}
    return {}


def should_disable_stateful_guard(
    *,
    model: str,
    decoding_config: "DecodingConfig",
) -> bool:
    return model == _QWEN_CHAT_MODEL and decoding_config.algorithm == "contrastive"


@dataclass(frozen=True)
class DecodingConfig:
    algorithm: str
    num_beams: int | None = None
    penalty_alpha: float | None = None
    top_k: int | None = None
    temperature: float = 0.0

    def validate(self) -> None:
        if self.temperature < 0.0:
            raise ValueError("HF local inference temperature must be non-negative.")
        if self.algorithm == "greedy":
            if self.num_beams is not None or self.penalty_alpha is not None:
                raise ValueError("Greedy decoding cannot set beam or contrastive parameters.")
            return
        if self.algorithm == "beam":
            if self.num_beams is None or self.num_beams < 2:
                raise ValueError("Beam search requires num_beams >= 2.")
            if self.penalty_alpha is not None or self.top_k is not None:
                raise ValueError("Beam search cannot set contrastive parameters.")
            return
        if self.algorithm == "contrastive":
            if self.penalty_alpha is None:
                raise ValueError("Contrastive search requires penalty_alpha.")
            if self.top_k is None:
                raise ValueError("Contrastive search requires top_k.")
            if self.num_beams is not None:
                raise ValueError("Contrastive search cannot set num_beams.")
            return
        raise ValueError(f"Unsupported decoding algorithm: {self.algorithm}")


def build_default_decoding_grid() -> list[DecodingConfig]:
    return [
        DecodingConfig(algorithm="greedy"),
        DecodingConfig(algorithm="beam", num_beams=2),
        DecodingConfig(algorithm="beam", num_beams=6),
        DecodingConfig(algorithm="contrastive", penalty_alpha=0.2, top_k=4),
        DecodingConfig(algorithm="contrastive", penalty_alpha=0.8, top_k=4),
    ]


def derive_context_window(
    *,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    safety_margin_tokens: int = 64,
) -> int:
    prompt_token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
    return derive_context_window_from_input_length(
        tokenizer=tokenizer,
        input_token_count=prompt_token_count,
        max_new_tokens=max_new_tokens,
        safety_margin_tokens=safety_margin_tokens,
    )


def derive_context_window_from_input_length(
    *,
    tokenizer: Any,
    input_token_count: int,
    max_new_tokens: int,
    safety_margin_tokens: int = 64,
) -> int:
    required_window = input_token_count + max_new_tokens + safety_margin_tokens
    max_model_length = _resolve_context_limit(tokenizer)
    if max_model_length is not None and required_window > max_model_length:
        raise ValueError(
            "Prompt requires "
            f"{required_window} tokens but tokenizer limit is {max_model_length}."
        )
    return required_window


@dataclass(frozen=True)
class RuntimeProfile:
    device: str
    dtype: str
    quantization: str
    supports_thinking_toggle: bool
    context_limit: int | None


class HFTransformersChatClient:
    def __init__(
        self,
        *,
        model: str,
        decoding_config: DecodingConfig,
        tokenizer: Any,
        model_object: Any,
        device: str,
        thinking_mode_supported: bool = False,
        max_new_tokens_by_schema: dict[str, int] | None = None,
        context_policy: dict[str, object] | None = None,
        seed: int = 0,
    ) -> None:
        decoding_config.validate()
        if not supports_decoding_config(model=model, decoding_config=decoding_config):
            raise ValueError(
                f"Decoding algorithm {decoding_config.algorithm!r} is unsupported for model"
                f" {model}."
            )
        self._model = model
        self._decoding_config = decoding_config
        self._tokenizer = tokenizer
        self._model_object = model_object
        self._device = device
        self._seed = seed
        self.thinking_mode_supported = thinking_mode_supported
        self._context_policy = context_policy or {}
        self.last_call_metrics: dict[str, object] | None = None
        self.last_failed_response_text: str | None = None
        self._max_new_tokens_by_schema = max_new_tokens_by_schema or {
            "edge_list": 256,
            "vote_list": 64,
            "label_list": 128,
            "children_by_label": 384,
        }
        _set_generation_temperature(model_object=self._model_object, temperature=0.0)

    def complete_json(
        self,
        *,
        prompt: str,
        schema_name: str,
        schema: dict[str, object],
    ) -> dict[str, object]:
        _ = schema
        max_new_tokens = self._max_new_tokens_by_schema.get(schema_name, 256)
        safety_margin_tokens = _resolve_safety_margin_tokens(self._context_policy)
        malformed_output_retries = 0
        malformed_output_retry_limit = _resolve_malformed_output_retry_limit(
            model=self._model,
            decoding_config=self._decoding_config,
            schema_name=schema_name,
        )
        encoded_inputs = _encode_chat_prompt(
            tokenizer=self._tokenizer,
            prompt=prompt,
            model=self._model,
            device=self._device,
            disable_thinking=self.thinking_mode_supported,
        )
        input_length = int(encoded_inputs["input_ids"].shape[-1])
        eos_token_id = getattr(self._tokenizer, "eos_token_id", None)

        while True:
            _ = derive_context_window_from_input_length(
                tokenizer=self._tokenizer,
                input_token_count=input_length,
                max_new_tokens=max_new_tokens,
                safety_margin_tokens=safety_margin_tokens,
            )
            generation_kwargs = {
                **encoded_inputs,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": _resolve_pad_token_id(self._tokenizer),
                "eos_token_id": eos_token_id,
                "use_cache": True,
                **_decoding_kwargs(self._decoding_config, model=self._model),
            }
            generation_kwargs.update(
                runtime_generation_overrides(
                    model=self._model,
                    decoding_config=self._decoding_config,
                )
            )
            generation_kwargs.update(
                _custom_generate_overrides(
                    model=self._model,
                    model_object=self._model_object,
                    decoding_config=self._decoding_config,
                )
            )
            generation_timeout_seconds = _resolve_generation_timeout_seconds(
                self._context_policy
            )
            if (
                generation_timeout_seconds is not None
                and self._decoding_config.algorithm != "contrastive"
            ):
                generation_kwargs["max_time"] = generation_timeout_seconds

            retry_seed = self._seed + malformed_output_retries
            torch = _torch()
            torch.manual_seed(retry_seed)
            random.seed(retry_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(retry_seed)
            generator = torch.Generator(device=self._device).manual_seed(retry_seed)
            generation_kwargs["generator"] = generator
            started_at = time.perf_counter()
            with torch.inference_mode():
                try:
                    with _temporarily_disable_stateful_guard(
                        model=self._model,
                        model_object=self._model_object,
                        decoding_config=self._decoding_config,
                    ):
                        generated_ids = self._model_object.generate(**generation_kwargs)
                except ValueError as error:
                    if not _generator_kwarg_rejected(error):
                        raise
                    generation_kwargs.pop("generator", None)
                    with _temporarily_disable_stateful_guard(
                        model=self._model,
                        model_object=self._model_object,
                        decoding_config=self._decoding_config,
                    ):
                        generated_ids = self._model_object.generate(**generation_kwargs)
            duration_seconds = time.perf_counter() - started_at
            completion_ids = generated_ids[0][input_length:]
            text = self._tokenizer.decode(completion_ids, skip_special_tokens=True)
            try:
                parsed_content = _parse_generated_json(text, schema_name=schema_name)
            except ValueError as error:
                self.last_failed_response_text = text
                if _response_hit_generation_limit(
                    completion_ids=completion_ids,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=eos_token_id,
                ):
                    max_new_tokens = _next_max_new_tokens(
                        tokenizer=self._tokenizer,
                        input_token_count=input_length,
                        current_max_new_tokens=max_new_tokens,
                        safety_margin_tokens=safety_margin_tokens,
                    )
                    continue
                if (
                    malformed_output_retries < malformed_output_retry_limit
                    and _looks_retryable_malformed_output(text=text, schema_name=schema_name)
                ):
                    max_new_tokens = _next_max_new_tokens(
                        tokenizer=self._tokenizer,
                        input_token_count=input_length,
                        current_max_new_tokens=max_new_tokens,
                        safety_margin_tokens=safety_margin_tokens,
                    )
                    malformed_output_retries += 1
                    continue
                if _should_normalize_exhausted_malformed_edge_list_to_empty(
                    model=self._model,
                    decoding_config=self._decoding_config,
                    schema_name=schema_name,
                    malformed_output_retries=malformed_output_retries,
                    malformed_output_retry_limit=malformed_output_retry_limit,
                    text=text,
                ):
                    normalized_content = {"edges": []}
                    self.last_failed_response_text = None
                    self.last_call_metrics = {
                        "schema_name": schema_name,
                        "prompt_token_count": input_length,
                        "completion_token_count": len(completion_ids),
                        "max_new_tokens": max_new_tokens,
                        "duration_seconds": round(duration_seconds, 6),
                        "tokens_per_second": round(
                            len(completion_ids) / duration_seconds,
                            6,
                        )
                        if duration_seconds > 0
                        else None,
                    }
                    return normalized_content
                raise error
            try:
                normalized_content = normalize_structured_response(
                    parsed_content,
                    schema_name=schema_name,
                )
            except ValueError as error:
                self.last_failed_response_text = text
                if (
                    malformed_output_retries < malformed_output_retry_limit
                    and _looks_retryable_normalization_failure(
                        parsed_content=parsed_content,
                        schema_name=schema_name,
                        error=error,
                    )
                ):
                    malformed_output_retries += 1
                    continue
                raise error
            self.last_failed_response_text = None
            completion_token_count = len(completion_ids)
            self.last_call_metrics = {
                "schema_name": schema_name,
                "prompt_token_count": input_length,
                "completion_token_count": completion_token_count,
                "max_new_tokens": max_new_tokens,
                "duration_seconds": round(duration_seconds, 6),
                "tokens_per_second": round(
                    completion_token_count / duration_seconds,
                    6,
                )
                if duration_seconds > 0
                else None,
            }
            return normalized_content


class HFTransformersEmbeddingClient:
    def __init__(
        self,
        *,
        tokenizer: Any,
        model: Any,
        device: str,
    ) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._device = device

    def embed_texts(self, texts: list[str]) -> dict[str, list[float]]:
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}

        torch = _torch()
        with torch.inference_mode():
            outputs = self._model(**encoded)
        hidden_state = outputs.last_hidden_state.detach().cpu().float().numpy()
        attention_mask = encoded["attention_mask"].detach().cpu().numpy()[..., None]
        masked = hidden_state * attention_mask
        pooled = masked.sum(axis=1) / np.clip(attention_mask.sum(axis=1), 1, None)
        return {text: pooled[index].astype(float).tolist() for index, text in enumerate(texts)}


class HFTransformersRuntimeFactory:
    def __init__(self, *, hf_token: str | None = None) -> None:
        self._hf_token = hf_token
        self._chat_cache: dict[str, tuple[Any, Any, RuntimeProfile]] = {}
        self._embedding_cache: dict[str, tuple[Any, Any, RuntimeProfile]] = {}

    def build_chat_client(
        self,
        *,
        model: str,
        decoding_config: DecodingConfig,
        max_new_tokens_by_schema: dict[str, int] | None = None,
        context_policy: dict[str, object] | None = None,
        seed: int = 0,
    ) -> HFTransformersChatClient:
        tokenizer, model_object, profile = self._load_chat_bundle(model)
        return HFTransformersChatClient(
            model=model,
            decoding_config=decoding_config,
            tokenizer=tokenizer,
            model_object=model_object,
            device=profile.device,
            thinking_mode_supported=profile.supports_thinking_toggle,
            max_new_tokens_by_schema=max_new_tokens_by_schema,
            context_policy=context_policy,
            seed=seed,
        )

    def build_embedding_client(self, *, model: str) -> HFTransformersEmbeddingClient:
        tokenizer, model_object, profile = self._load_embedding_bundle(model)
        return HFTransformersEmbeddingClient(
            tokenizer=tokenizer,
            model=model_object,
            device=profile.device,
        )

    def profile_for_chat_model(self, model: str) -> RuntimeProfile:
        _, _, profile = self._load_chat_bundle(model)
        return profile

    def prefetch_models(
        self,
        *,
        chat_models: list[str],
        embedding_model: str,
    ) -> dict[str, object]:
        prefetched_chat_models: list[str] = []
        for chat_model in chat_models:
            self._load_chat_bundle(chat_model)
            prefetched_chat_models.append(chat_model)
            self._release_prefetched_chat_bundle(chat_model)
        self._load_embedding_bundle(embedding_model)
        self._release_prefetched_embedding_bundle(embedding_model)
        return {
            "chat_models": prefetched_chat_models,
            "embedding_model": embedding_model,
        }

    def _release_prefetched_chat_bundle(self, model: str) -> None:
        bundle = self._chat_cache.pop(model, None)
        if bundle is None:
            return
        _, model_object, _ = bundle
        _release_prefetched_model_object(model_object)

    def _release_prefetched_embedding_bundle(self, model: str) -> None:
        bundle = self._embedding_cache.pop(model, None)
        if bundle is None:
            return
        _, model_object, _ = bundle
        _release_prefetched_model_object(model_object)

    def _load_chat_bundle(self, model: str) -> tuple[Any, Any, RuntimeProfile]:
        if model not in _SUPPORTED_CHAT_MODELS:
            raise ValueError(f"Unsupported hf-transformers chat model: {model}")
        cached = self._chat_cache.get(model)
        if cached is not None:
            return cached

        transformers = _transformers()
        torch = _torch()
        _require_cuda(torch)
        tokenizer = _load_chat_tokenizer(
            transformers=transformers,
            model=model,
            hf_token=self._hf_token,
        )
        profile = _build_runtime_profile(
            model,
            context_limit=_resolve_context_limit(tokenizer),
        )
        model_object = _load_chat_model(
            transformers=transformers,
            model=model,
            hf_token=self._hf_token,
            torch_dtype=_dtype_from_profile(torch, profile.dtype),
        )
        if not _model_uses_accelerate_device_map(model_object):
            model_object.to(profile.device)
        model_object.eval()
        bundle = (tokenizer, model_object, profile)
        self._chat_cache[model] = bundle
        return bundle

    def _load_embedding_bundle(self, model: str) -> tuple[Any, Any, RuntimeProfile]:
        if model not in _SUPPORTED_EMBEDDING_MODELS:
            raise ValueError(f"Unsupported hf-transformers embedding model: {model}")
        cached = self._embedding_cache.get(model)
        if cached is not None:
            return cached

        transformers = _transformers()
        torch = _torch()
        _require_cuda(torch)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, token=self._hf_token)
        profile = _build_runtime_profile(
            model,
            context_limit=_resolve_context_limit(tokenizer),
        )
        model_object = transformers.AutoModel.from_pretrained(
            model,
            token=self._hf_token,
            torch_dtype=_dtype_from_profile(torch, profile.dtype),
            attn_implementation="sdpa",
        )
        model_object.to(profile.device)
        model_object.eval()
        bundle = (tokenizer, model_object, profile)
        self._embedding_cache[model] = bundle
        return bundle


def build_runtime_factory(*, hf_token: str | None = None) -> HFTransformersRuntimeFactory:
    return HFTransformersRuntimeFactory(hf_token=hf_token)


def _release_prefetched_model_object(model_object: Any) -> None:
    try:
        model_object.to("cpu")
    except Exception:
        pass
    del model_object
    gc.collect()
    try:
        torch = _torch()
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()


def _model_uses_accelerate_device_map(model_object: Any) -> bool:
    device_map = getattr(model_object, "hf_device_map", None)
    return isinstance(device_map, dict) and bool(device_map)


def _load_chat_tokenizer(*, transformers: Any, model: str, hf_token: str | None) -> Any:
    remote_code_kwargs = _trusted_remote_code_kwargs(model)
    if model == _MINISTRAL_CHAT_MODEL and hasattr(transformers, "MistralCommonBackend"):
        return transformers.MistralCommonBackend.from_pretrained(
            model,
            token=hf_token,
            **remote_code_kwargs,
        )
    return transformers.AutoTokenizer.from_pretrained(
        model,
        token=hf_token,
        **remote_code_kwargs,
    )


def _load_chat_model(
    *,
    transformers: Any,
    model: str,
    hf_token: str | None,
    torch_dtype: Any,
) -> Any:
    remote_code_kwargs = _trusted_remote_code_kwargs(model)
    if model == _MINISTRAL_CHAT_MODEL and hasattr(
        transformers, "Mistral3ForConditionalGeneration"
    ):
        fp8_config = transformers.FineGrainedFP8Config(dequantize=True)
        return transformers.Mistral3ForConditionalGeneration.from_pretrained(
            model,
            token=hf_token,
            device_map="auto",
            dtype=torch_dtype,
            quantization_config=fp8_config,
            **remote_code_kwargs,
        )
    return transformers.AutoModelForCausalLM.from_pretrained(
        model,
        token=hf_token,
        dtype=torch_dtype,
        attn_implementation=_resolve_attention_implementation(model),
        **remote_code_kwargs,
    )


def _trusted_remote_code_kwargs(model: str) -> dict[str, object]:
    if model in _TRUSTED_REMOTE_CODE_CHAT_MODELS:
        return {"trust_remote_code": True}
    return {}


def _resolve_attention_implementation(model: str) -> str:
    if model == "allenai/Olmo-3-7B-Instruct" and _flash_attention_available():
        return "flash_attention_2"
    return "sdpa"


@lru_cache(maxsize=1)
def _flash_attention_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def _decoding_kwargs(config: DecodingConfig, *, model: str | None = None) -> dict[str, object]:
    if config.algorithm == "greedy":
        return {"do_sample": False}
    if config.algorithm == "beam":
        return {"do_sample": False, "num_beams": config.num_beams}
    kwargs = {
        "do_sample": False,
        "penalty_alpha": config.penalty_alpha,
        "top_k": config.top_k,
    }
    if model in _TRUSTED_REMOTE_CODE_CHAT_MODELS:
        kwargs["trust_remote_code"] = True
    return kwargs


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
    qwen_dynamic_cache_type = _load_qwen_dynamic_cache_type()
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
    except Exception:
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


def _parse_generated_json(text: str, *, schema_name: str) -> object:
    stripped = _strip_code_fence(_strip_assistant_prefix(text.strip()))
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        recovered = _recover_non_json_response(text=stripped, schema_name=schema_name)
        if recovered is not None:
            return recovered
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError) as error:
            raise ValueError(f"Model did not return valid structured output: {text}") from error
    return _normalize_schema_response(parsed, schema_name=schema_name)


def _strip_assistant_prefix(text: str) -> str:
    lowered = text.lower()
    for prefix in ("assistant\n", "assistant:\n", "assistant: ", "assistant "):
        if lowered.startswith(prefix):
            return text[len(prefix) :].strip()
    if lowered.startswith("assistant"):
        suffix = text[len("assistant") :].strip()
        if suffix and all(not character.isalnum() for character in suffix):
            return ""
    return text


def _strip_code_fence(text: str) -> str:
    """Strip markdown code fence markers from text.

    Uses non-greedy .*? to find the first ``` closing fence. If the
    extracted body has an unclosed JSON string (odd quote count), the
    closing fence was embedded inside a string key/value. In that case,
    re-extracts using the LAST ``` as the true closing fence.
    """
    fenced = re.search(r"```(?P<lang>[A-Za-z0-9_-]*)\s*(?P<body>.*?)```", text, flags=re.DOTALL)
    if fenced is None:
        return text
    body = fenced.group("body").strip()
    # If body has odd quote count, the closing fence was embedded inside a JSON
    # string. Re-extract using the LAST ``` as the true closing fence.
    if body.count('"') % 2 == 1:
        last_fence_pos = text.rfind("```")
        if last_fence_pos > fenced.start():
            # Skip past the opening fence lang line (```lang\n)
            open_lang_end = text.find("\n", fenced.start())
            if open_lang_end < 0:
                open_lang_end = fenced.end()
            else:
                open_lang_end += 1  # include the newline
            body = text[open_lang_end:last_fence_pos].strip()
    return body


def _normalize_schema_response(parsed: object, *, schema_name: str) -> object:
    if isinstance(parsed, str):
        recovered = _recover_non_json_response(text=parsed, schema_name=schema_name)
        if recovered is not None:
            return recovered
    if schema_name == "label_list":
        recovered_labels = _normalize_label_list_payload(parsed)
        if recovered_labels is not None:
            return recovered_labels
    if schema_name == "children_by_label" and _looks_like_children_mapping(parsed):
        return {"children_by_label": parsed}
    return parsed


def _looks_retryable_malformed_output(*, text: str, schema_name: str) -> bool:
    stripped = _strip_code_fence(_strip_assistant_prefix(text.strip()))
    if not stripped:
        return True
    lowered = stripped.lower()
    if "<think>" in lowered or "</think>" in lowered:
        return True
    if schema_name == "edge_list":
        if stripped.startswith("[") and not stripped.rstrip().endswith("]"):
            return True
        quoted_items = re.findall(r"""['"]([^'"]+)['"]""", stripped)
        if quoted_items and len(quoted_items) % 2 == 1:
            return True
    if schema_name == "children_by_label":
        if stripped.startswith("{") and not stripped.rstrip().endswith("}"):
            return True
    return False


def _looks_retryable_normalization_failure(
    *,
    parsed_content: object,
    schema_name: str,
    error: ValueError,
) -> bool:
    if schema_name != "edge_list":
        return False
    if "even number of items" not in str(error):
        return False
    return isinstance(parsed_content, list) and all(
        isinstance(item, str) for item in parsed_content
    )


def _resolve_malformed_output_retry_limit(
    *,
    model: str,
    decoding_config: DecodingConfig,
    schema_name: str,
) -> int:
    if (
        model == _QWEN_CHAT_MODEL
        and decoding_config.algorithm == "contrastive"
        and schema_name in {"edge_list", "children_by_label"}
    ):
        return 3
    return 1


def _should_normalize_exhausted_malformed_edge_list_to_empty(
    *,
    model: str,
    decoding_config: DecodingConfig,
    schema_name: str,
    malformed_output_retries: int,
    malformed_output_retry_limit: int,
    text: str,
) -> bool:
    if model != _QWEN_CHAT_MODEL:
        return False
    if decoding_config.algorithm != "contrastive":
        return False
    if schema_name != "edge_list":
        return False
    if malformed_output_retries < malformed_output_retry_limit:
        return False
    return _looks_like_truncated_single_edge_endpoint(text)


def _normalize_label_list_payload(parsed: object) -> list[str] | None:
    if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
        return cast(list[str], parsed)
    if not isinstance(parsed, Mapping):
        return None
    parsed_mapping = cast(Mapping[str, object], parsed)
    labels = parsed_mapping.get("labels")
    if isinstance(labels, list) and all(isinstance(item, str) for item in labels):
        string_labels = cast(list[str], labels)
        if len(labels) == 1:
            packed_labels = _split_packed_label_string(string_labels[0])
            if packed_labels is not None:
                return packed_labels
        return string_labels
    if isinstance(labels, str):
        return _split_packed_label_string(labels)
    return None


def _split_packed_label_string(text: str) -> list[str] | None:
    if "', '" not in text and '", "' not in text:
        stripped = text.strip().strip("'\"")
        return [stripped] if stripped else None
    parts = re.split(r"""['"]\s*,\s*['"]""", text.strip())
    cleaned_parts = [part.strip().strip("'\"") for part in parts if part.strip().strip("'\"")]
    if not cleaned_parts:
        return None
    return cleaned_parts


def _looks_like_children_mapping(parsed: object) -> bool:
    if not isinstance(parsed, dict) or "children_by_label" in parsed:
        return False
    if not parsed:
        return False
    for key, value in parsed.items():
        if not isinstance(key, str):
            return False
        if not isinstance(value, list):
            return False
        if not all(isinstance(item, str) for item in value):
            return False
    return True


def _recover_non_json_response(*, text: str, schema_name: str) -> object | None:
    # Handle degenerate non-JSON outputs that contain no useful structure.
    stripped = text.strip().lower()
    if schema_name == "children_by_label":
        # Model returned bare markdown code fence (```json) or literal "Error".
        if not stripped or stripped in ("```json", "error", '"""json"""') or "```json" in stripped:
            return {"children_by_label": {}}
        candidate_texts = [text]
        # Strip thinking blocks, markdown bold markers, and embedded fences FIRST.
        # These are common model artifacts that corrupt JSON parsing before any recovery runs.
        artifact_stripped = _strip_fenced_content_artifacts(text)
        if artifact_stripped != text:
            candidate_texts.insert(0, artifact_stripped)
            # Also sanitize the artifact-stripped version
            sanitized_artifact = _sanitize_children_mapping_text_for_recovery(artifact_stripped)
            if sanitized_artifact != artifact_stripped and sanitized_artifact not in candidate_texts:
                candidate_texts.insert(1, sanitized_artifact)
        sanitized_text = _sanitize_children_mapping_text_for_recovery(text)
        if sanitized_text != text and sanitized_text not in candidate_texts:
            candidate_texts.append(sanitized_text)
        # Additional sanitization: remove non-string bracket patterns like [diet] reminders
        # or [exhaustion', low vitality'] that break the parser.
        extra_sanitized = _remove_nonstring_bracket_patterns(sanitized_text)
        if extra_sanitized != sanitized_text and extra_sanitized not in candidate_texts:
            candidate_texts.append(extra_sanitized)
        # Also handle trailing comma before } like ['Patience', 'Consistenc', }]
        comma_fixed = re.sub(r",\s*[\\]}]", "]", sanitized_text).rstrip()
        if comma_fixed != sanitized_text and comma_fixed not in candidate_texts:
            candidate_texts.append(comma_fixed)
        for candidate_text in candidate_texts:
            # Try fenced python children mapping FIRST (Mistral pattern with keys in quotes)
            # This needs to run before _remove_nonstring_bracket_patterns corrupts the text
            recovered_children = _recover_fenced_python_children_mapping(candidate_text)
            if recovered_children is not None:
                return {"children_by_label": recovered_children}
            if candidate_text.count("{") <= 1 and candidate_text.count("}") <= 1:
                recovered_children = _recover_double_quoted_children_values(candidate_text)
                if recovered_children is not None:
                    return {"children_by_label": recovered_children}
            recovered_children = _recover_children_mapping_from_outer_block(candidate_text)
            if recovered_children is not None:
                return {"children_by_label": recovered_children}
            recovered_children = _recover_malformed_children_mapping(candidate_text)
            if recovered_children is not None:
                return {"children_by_label": recovered_children}
            recovered_children = _recover_inline_children_mapping(candidate_text)
            if recovered_children is not None:
                return {"children_by_label": recovered_children}
            recovered_children = _recover_children_mapping_from_lines(candidate_text)
            if recovered_children is not None:
                return {"children_by_label": recovered_children}
            # Last resort: try unquoted key with comma-separated values
            recovered_children = _recover_unquoted_key_comma_separated(candidate_text)
            if recovered_children is not None:
                return {"children_by_label": recovered_children}
        # Also try truncated candidates for patterns like {Key: [val1, val2}
        truncated_candidates = _recover_truncated_children_mapping_blocks(text)
        for candidate_text in truncated_candidates:
            if candidate_text == text:
                continue  # already tried
            recovered_children = _recover_fenced_python_children_mapping(candidate_text)
            if recovered_children is not None:
                return {"children_by_label": recovered_children}
            recovered_children = _recover_inline_children_mapping(candidate_text)
            if recovered_children is not None:
                return {"children_by_label": recovered_children}
    if schema_name == "label_list":
        recovered_labels = _recover_label_list_from_lines(text)
        if recovered_labels is not None:
            return recovered_labels
        recovered_labels = _recover_bare_comma_separated_label_list(text)
        if recovered_labels is not None:
            return recovered_labels
        recovered_labels = _recover_quoted_label_list_with_comments(text)
        if recovered_labels is not None:
            return recovered_labels
        recovered_labels = _recover_single_bare_label(text)
        if recovered_labels is not None:
            return recovered_labels
    if schema_name == "edge_list":
        recovered_edge_pairs = _recover_bracketed_edge_pairs(text)
        if recovered_edge_pairs is not None:
            return recovered_edge_pairs
        recovered_edge_pairs = _recover_bare_comma_separated_edge_pair(text)
        if recovered_edge_pairs is not None:
            return recovered_edge_pairs
        tuple_matches = re.findall(r"\(([^()]*)\)", text)
        if tuple_matches:
            parsed_edges: list[tuple[str, str]] = []
            for tuple_text in tuple_matches:
                parts = [part.strip().strip("'\"") for part in tuple_text.split(",", 1)]
                if len(parts) != 2 or not parts[0] or not parts[1]:
                    continue
                parsed_edges.append((parts[0], parts[1]))
            if parsed_edges:
                return parsed_edges
        quoted_endpoints = _extract_recoverable_edge_endpoints(text)
        if quoted_endpoints is not None:
            return quoted_endpoints
    if schema_name == "vote_list":
        token_matches = re.findall(r"\b[YyNn]\b", text)
        if token_matches:
            return [token.upper() for token in token_matches]
    return None


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


def _recover_malformed_children_mapping(text: str) -> dict[str, list[str]] | None:
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    candidates = [f"{stripped}}}"]
    if stripped.endswith("]"):
        candidates.append(f"{stripped[:-1]}}}")
    for candidate in candidates:
        try:
            parsed = ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            continue
        if _looks_like_children_mapping(parsed):
            return parsed
    return None


def _recover_children_mapping_from_outer_block(text: str) -> dict[str, list[str]] | None:
    first_balanced = _extract_first_balanced_block(text, opener="{", closer="}")
    outer_block = _extract_outer_block(text=text, opener="{", closer="}")
    blocks_to_try: list[str] = []
    if first_balanced is not None:
        blocks_to_try.append(first_balanced)
    if outer_block is not None and outer_block != first_balanced:
        blocks_to_try.append(outer_block)
    if not blocks_to_try:
        return None
    for block in blocks_to_try:
        candidates = [block]
        stripped_comments = _strip_mapping_comments(block)
        if stripped_comments != block:
            candidates.append(stripped_comments)
        for candidate in candidates:
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(candidate)
                except (json.JSONDecodeError, ValueError, SyntaxError):
                    continue
                if isinstance(parsed, dict) and not parsed:
                    return {}
                if _looks_like_children_mapping(parsed):
                    return parsed
    return None


def _strip_mapping_comments(text: str) -> str:
    without_block_comments = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    without_inline_comments = re.sub(r"//[^\n]*", "", without_block_comments)
    without_paren_notes = re.sub(
        r"\(\s*(?:Note|WARNING|Caveat|Correction|Actually)\b[^)]*\)",
        "",
        without_inline_comments,
        flags=re.IGNORECASE,
    )
    without_orphan_paren_lines = re.sub(
        r"(?m)^\s*\([^()]*\),?\s*$",
        "",
        without_paren_notes,
    )
    result: list[str] = []
    index = 0
    in_single_quote = False
    in_double_quote = False
    while index < len(without_orphan_paren_lines):
        character = without_orphan_paren_lines[index]
        if character == "\\":
            result.append(character)
            index += 1
            if index < len(without_orphan_paren_lines):
                result.append(without_orphan_paren_lines[index])
            index += 1
            continue
        if character == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            result.append(character)
            index += 1
            continue
        if character == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            result.append(character)
            index += 1
            continue
        if character == "#" and not in_single_quote and not in_double_quote:
            while index < len(without_orphan_paren_lines) and without_orphan_paren_lines[index] != "\n":
                index += 1
            continue
        result.append(character)
        index += 1
    return "".join(result)


def _recover_inline_children_mapping(text: str) -> dict[str, list[str]] | None:
    first_balanced = _extract_first_balanced_block(text, opener="{", closer="}")
    outer_block = _extract_outer_block(text=text, opener="{", closer="}")
    blocks_to_try: list[str] = []
    if first_balanced is not None:
        blocks_to_try.append(first_balanced)
    if outer_block is not None and outer_block != first_balanced:
        blocks_to_try.append(outer_block)
    if not blocks_to_try:
        blocks_to_try.extend(_recover_truncated_children_mapping_blocks(text))
    if not blocks_to_try:
        return None
    for block in blocks_to_try:
        result = _try_inline_children_parse(block)
        if result is not None:
            return result
    return None


def _try_inline_children_parse(block: str) -> dict[str, list[str]] | None:
    mapping: dict[str, list[str]] = {}
    index = 1
    length = len(block)
    while index < length:
        index = _skip_inline_mapping_separators(block, index)
        if index >= length or block[index] == "}":
            break
        key_result = _scan_lenient_mapping_key(block, index)
        if key_result is None:
            return None
        key, index = key_result
        index = _skip_inline_mapping_separators(block, index)
        if index >= length or block[index] != ":":
            return None
        index += 1
        index = _skip_inline_mapping_separators(block, index)
        if index >= length or block[index] != "[":
            return None
        values_result = _scan_lenient_quoted_list(block, index)
        if values_result is None:
            return None
        values, index = values_result
        existing_values = mapping.get(key)
        if existing_values and not values:
            pass
        else:
            mapping[key] = values
        index = _skip_inline_mapping_separators(block, index)
        if index < length and block[index] == ",":
            index += 1
    if not mapping:
        return None
    return mapping


def _skip_inline_mapping_separators(text: str, index: int) -> int:
    while index < len(text) and text[index] in {" ", "\t", "\n", "\r"}:
        index += 1
    return index


def _scan_lenient_mapping_key(text: str, start_index: int) -> tuple[str, int] | None:
    quoted_result = _scan_lenient_quoted_string(text, start_index)
    if quoted_result is not None:
        return quoted_result
    return _scan_unquoted_mapping_key(text, start_index)


def _scan_lenient_quoted_list(text: str, start_index: int) -> tuple[list[str], int] | None:
    if text[start_index] != "[":
        return None
    index = start_index + 1
    values: list[str] = []
    while index < len(text):
        index = _skip_inline_mapping_separators(text, index)
        if index >= len(text):
            return None
        if text[index] in {"]", ")"}:
            return values, index + 1
        if text[index] in {"'", '"'}:
            next_index = _skip_inline_mapping_separators(text, index + 1)
            if next_index < len(text) and text[next_index] in {"]", "}", ")"}:
                return values, next_index + 1
        if text[index] == "[":
            index = _skip_nested_bracketed_value(text, index)
            if index == -1:
                return None
            index = _skip_inline_mapping_separators(text, index)
            if index < len(text) and text[index] == ",":
                index += 1
                continue
            if index < len(text) and text[index] in {"]", ")"}:
                return values, index + 1
            return None
        item_result = _scan_lenient_quoted_string(text, index)
        if item_result is None:
            item_result = _scan_unquoted_list_item(text, index)
        if item_result is None:
            return None
        item, index = item_result
        values.append(item)
        index = _skip_inline_mapping_separators(text, index)
        if index < len(text) and text[index] == ",":
            index += 1
            continue
        if index < len(text) and text[index] in {"]", ")"}:
            return values, index + 1
        return None
    return None


def _skip_nested_bracketed_value(text: str, start_index: int) -> int:
    if start_index >= len(text) or text[start_index] != "[":
        return -1
    depth = 0
    index = start_index
    while index < len(text):
        character = text[index]
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return -1


def _scan_lenient_quoted_string(text: str, start_index: int) -> tuple[str, int] | None:
    if start_index >= len(text) or text[start_index] not in {"'", '"'}:
        return None
    opening_quote = text[start_index]
    delimiter_chars = {",", "]", "}", ":", ")"}
    index = start_index + 1
    while index < len(text):
        current_char = text[index]
        if current_char in {"'", '"'}:
            next_index = _skip_inline_mapping_separators(text, index + 1)
            if next_index >= len(text) or text[next_index] in delimiter_chars:
                if current_char != opening_quote:
                    matching_quote_index = text.find(opening_quote, index + 1)
                    if matching_quote_index != -1:
                        index += 1
                        continue
                value = text[start_index + 1 : index].strip()
                return value, index + 1
            if next_index < len(text) and text[next_index] in {"'", '"'}:
                next_next_index = _skip_inline_mapping_separators(text, next_index + 1)
                if next_next_index < len(text) and text[next_next_index] in delimiter_chars:
                    if current_char == opening_quote:
                        value = text[start_index + 1 : index].strip()
                        return value, next_index + 1
            if (
                current_char == opening_quote
                and index > start_index + 1
                and index + 1 < len(text)
                and text[index - 1].isalnum()
                and text[index + 1].isalnum()
            ):
                index += 1
                continue
        index += 1
    return None


def _scan_unquoted_list_item(text: str, start_index: int) -> tuple[str, int] | None:
    """Scan a list item that lacks an opening quote."""
    delimiter_chars = {",", "]", "}", ")"}
    index = start_index
    while index < len(text):
        current_char = text[index]
        if current_char in delimiter_chars:
            value = text[start_index:index].strip()
            if value:
                return value, index
            return None
        if current_char in {"'", '"'}:
            next_index = _skip_inline_mapping_separators(text, index + 1)
            if next_index >= len(text) or text[next_index] in delimiter_chars:
                value = text[start_index:index].strip()
                if value:
                    return value, index + 1
                return None
        index += 1
    return None


def _scan_unquoted_mapping_key(text: str, start_index: int) -> tuple[str, int] | None:
    index = start_index
    while index < len(text) and text[index] != ":":
        if text[index] in {"}", "]", ","}:
            return None
        index += 1
    if index >= len(text) or text[index] != ":":
        return None
    value = text[start_index:index].strip().strip("{").strip()
    if not value:
        return None
    value = value.strip("'\"").strip()
    return value, index


def _sanitize_children_mapping_text_for_recovery(text: str) -> str:
    sanitized = text.strip()
    if sanitized.startswith('"') and not sanitized.endswith('"'):
        sanitized = sanitized[1:]
    if sanitized.endswith("\\"):
        sanitized = sanitized[:-1]
    # Fix Mistral artifact where outer quote is swapped with inner quote at item end
    # e.g., '"Thinspiration"', or '"Thinspiration"\', -> '"Thinspiration"' ,
    sanitized = re.sub(r"\\?'\"\s*([,\]\n])", '"' + "'" + r"\1", sanitized)
    sanitized = re.sub(
        r"\\u([0-9a-fA-F]{4})",
        lambda match: chr(int(match.group(1), 16)),
        sanitized,
    )
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\uff1a": ":",
    }
    for source, target in replacements.items():
        sanitized = sanitized.replace(source, target)
    return sanitized


def _strip_fenced_content_artifacts(text: str) -> str:
    """Strip model-generated artifacts from fenced code block content.

    Handles three common artifact families in Mistral/Qwen contrastive outputs:
    1. Embedded ``` inside JSON strings (e.g., '"Key\n```ng"') - strips the embedded fence
    2. <think>...</think> thinking block content - removes thinking text AND delimiters
    3. **bold** markdown markers - strips bold wrappers, keeps inner text

    These artifacts break JSON parsing and are not valid JSON structure.
    """
    # 1. Strip embedded fences inside strings: \n```lang (at end of a string value)
    result = re.sub(r'\n```[a-zA-z0-9]*', '', text)
    # 2. Strip <think>...</think> thinking blocks (content AND delimiters)
    result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL)
    # 3. Strip **bold** markdown markers, keep inner text
    result = re.sub(r"\*\*(.+?)\*\*", r"\1", result)
    # 4. After bold strip, trailing '** :' or '**:' may remain at end of line.
    #    Strip trailing '** :' and '**:' with their trailing quote/colons cleanly.
    #    Replace '** :' with ':' and '**:' with ':' at end of content lines.
    result = re.sub(r"\*\*\s*:\s*", ":", result)  # '** :' -> ':'
    result = re.sub(r"\*\*", "", result)  # remove remaining '**'
    return result


def _recover_unquoted_key_comma_separated(text: str) -> dict[str, list[str]] | None:
    """Recover from unquoted key with comma-separated values (no brackets).

    Handles: {KeyName: Value1, Value2, Value3}
    Returns: {"KeyName": ["Value1", "Value2", "Value3"]}
    """
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    # Skip texts that have brackets - those are handled by other recovery functions
    if "[" in stripped or "]" in stripped:
        return None

    # Find the first colon that separates key from values
    colon_pos = stripped.find(":")
    if colon_pos == -1:
        return None

    key = stripped[1:colon_pos].strip()
    values_part = stripped[colon_pos + 1:].strip().rstrip("}")

    # Split on commas to get individual values
    items = re.split(r",\s*", values_part)
    values = [v.strip().strip("'\"").strip() for v in items if v.strip()]

    if not key or not values:
        return None

    return {key: values}


def _recover_fenced_python_children_mapping(text: str) -> dict[str, list[str]] | None:
    """Recover from fenced python block or inline python with children mapping missing colons.

    Handles:
    ```python
    {
        "anorexia nervosa" ["starvation behavior", ...],
        "bulimiaproblematique" ["binge-eatingepisodestendency"]
    }
    ```
    Also handles text already stripped of fence markers.
    Returns: {"anorexia nervosa": ["starvation behavior", ...], ...}
    """
    stripped = text.strip()
    block = stripped

    # Try to extract fenced python block content
    match = re.search(r"```(?:\w+)?\s*(.*?)```", stripped, flags=re.DOTALL)
    if match:
        block = match.group(1).strip()
    elif stripped.startswith("```"):
        # Truncated fence - strip opening ``` and try to parse content
        block = re.sub(r"^```(?:\w+)?\s*", "", stripped).strip()

    if not block.startswith("{"):
        return None

    block = _strip_fenced_content_artifacts(block)

    # Handle truncated dict: ends with ] but missing closing }
    block = block.rstrip()
    if block.endswith("]") and not block.endswith("}]"):
        block = block + "}"

    quote_closed_candidate = re.sub(r"\s*:\s*\n\s*\[", "', ", block)
    if quote_closed_candidate != block:
        quoted_entry = _recover_first_quoted_children_entry(quote_closed_candidate)
        if quoted_entry is not None:
            return quoted_entry

    parsed_block = _try_inline_children_parse(block)
    if parsed_block is not None and not _children_mapping_needs_more_recovery(
        text=block, parsed=parsed_block
    ):
        return parsed_block

    block = _strip_mapping_comments(block)
    parsed_block = _try_inline_children_parse(block)
    if parsed_block is not None and not _children_mapping_needs_more_recovery(
        text=block, parsed=parsed_block
    ):
        return parsed_block

    # Join continuation lines (multi-line values) before parsing
    block = re.sub(r"\n\s+", " ", block)

    mapping: dict[str, list[str]] = {}
    for line in block.splitlines():
        line = line.strip().rstrip(",")
        # Strip leading dict artifacts but preserve key pattern
        line = re.sub(r"^[\s{,\\n]+", "", line)
        # Strip trailing dict artifacts but preserve ] at the end
        line = re.sub(r"[\s,]*(}[,\s]*)$", r"\1", line)
        if not line or line in {"}", "},"}:
            continue

        # Try fenced python patterns for Mistral children mapping without colons
        # These patterns handle: "Key" [...] or 'Key' [...] (no colon)
        # Also handle: "Key" (...) [...] or 'Key' (...) [...] (with parenthetical comment, no colon)
        # Pattern 1: double-quoted key (with or without comment), followed by space+bracket (no colon)
        kv_match = re.match(r'''"([^"]+)"(?:\s*\([^)]*\))?(?!\s*:)\s*\[([^\]]+)\]''', line)
        if kv_match:
            key = kv_match.group(1)
            values_str = kv_match.group(2)
            values = [v.strip().strip("'\"").strip('"') for v in values_str.split(",")]
            values = [v for v in values if v]
            if key and values:
                mapping[key] = values
                continue

        # Pattern 2: single-quoted key (with or without comment), followed by space+bracket (no colon)
        kv_match = re.match(r"""'([^']+)'(?:\s*\([^)]*\))?(?!\s*:)\s*\[([^\]]+)\]""", line)
        if kv_match:
            key = kv_match.group(1)
            values_str = kv_match.group(2)
            values = [v.strip().strip("'\"").strip('"') for v in values_str.split(",")]
            values = [v for v in values if v]
            if key and values:
                mapping[key] = values
                continue

        # Pattern 3: double-quoted key WITH parenthetical comment AND colon (Mistral specific)
        kv_match = re.match(r'''"([^"]+)"(?:\s*\([^)]*\)):\s*\[([^\]]+)\]''', line)
        if kv_match:
            key = kv_match.group(1)
            values_str = kv_match.group(2)
            values = [v.strip().strip("'\"").strip('"') for v in values_str.split(",")]
            values = [v for v in values if v]
            if key and values:
                mapping[key] = values
                continue

        # Pattern 4: single-quoted key WITH parenthetical comment AND colon (Mistral specific)
        kv_match = re.match(r"""'([^']+)'(?:\s*\([^)]*\)):\s*\[([^\]]+)\]""", line)
        if kv_match:
            key = kv_match.group(1)
            values_str = kv_match.group(2)
            values = [v.strip().strip("'\"").strip('"') for v in values_str.split(",")]
            values = [v for v in values if v]
            if key and values:
                mapping[key] = values
                continue

        # Pattern 5: outer double quotes with inner single-quoted key AND parenthetical comment (Mistral)
        # e.g., "'jun food dominance index' (JFD*)": ["value1", "value2"]
        # The key is: 'jun food dominance index' wrapped in double quotes
        _p5 = ('"' + "'" + "([^" + "'" + "]+)" + "'" +
               r"(?:\s*\([^)]*\))" + r":\s*\[" + r"([^\]]+)\]")
        kv_match = re.match(_p5, line)
        if kv_match:
            key = "'" + kv_match.group(1) + "'"
            values_str = kv_match.group(2)
            values = [v.strip().strip("'\"").strip('"') for v in values_str.split(",")]
            values = [v for v in values if v]
            if key and values:
                mapping[key] = values

    if mapping:
        return mapping

    first_entry = _recover_first_quoted_children_entry(block)
    if first_entry is not None and not _children_mapping_needs_more_recovery(
        text=block,
        parsed=first_entry,
    ):
        return first_entry
    return None


def _recover_first_quoted_children_entry(text: str) -> dict[str, list[str]] | None:
    patterns = (
        r"""'([^']+)'\s*:\s*\[(.*?)\]""",
        r'''"([^"]+)"\s*:\s*\[(.*?)\]''',
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.DOTALL)
        if match is None:
            continue
        key = match.group(1).strip()
        values_result = _scan_lenient_quoted_list(f"[{match.group(2)}]", 0)
        if values_result is None:
            continue
        values, _ = values_result
        if key and values:
            return {key: values}
    return None


def _recover_single_key_children_values(text: str) -> dict[str, list[str]] | None:
    match = re.search(r"""['"]([^'"]+)['"]\s*:\s*\[""", text, flags=re.DOTALL)
    if match is None:
        return None
    key = match.group(1).strip()
    if not key:
        return None
    values_text = text[match.end() :]
    values = [item.strip() for item in re.findall(r"""['"]([^'"]+)['"]""", values_text, flags=re.DOTALL)]
    values = [value for value in values if value]
    if not values:
        return None
    return {key: values}


def _recover_double_quoted_children_values(text: str) -> dict[str, list[str]] | None:
    match = re.search(r'"([^"]+)"\s*:\s*\[', text)
    if match is None:
        return None
    key = match.group(1).strip()
    if not key:
        return None
    values = [item.strip() for item in re.findall(r'"([^"]+)"', text[match.end() :])]
    if not values:
        return None
    return {key: values}


def _children_mapping_needs_more_recovery(
    *,
    text: str,
    parsed: Mapping[str, list[str]],
) -> bool:
    print(f"DEBUG: needs_more_recovery called with len={len(parsed)}")
    if len(parsed) != 1:
        return False
    values = next(iter(parsed.values()))
    if any(
        value.startswith(('"', "'"))
        or value.endswith(('"', "'"))
        or "[" in value
        or "]" in value
        or "\n" in value
        for value in values
    ):
        return True
    return bool(re.search(r"\[[^\]\"'\[]+\]", text))


def _remove_nonstring_bracket_patterns(text: str) -> str:
    """Remove non-string bracket patterns that break children_by_label recovery.

    Handles cases like:
    - '[diet] reminders'  -> removes [diet] and trailing garbage
    - "[exhaustion', low vitality']"  -> extracts valid strings
    """
    result = text
    # Remove bare [word] followed by space and more content
    result = re.sub(r"\[[^\"\[]+\][^\[\"]*", "", result)
    # Handle nested bracket patterns - extract quoted strings from within
    while True:
        before = result
        result = re.sub(
            r"\[([^'\"]*'[^'\"]*'[^'\"]*)'?\s*\]",
            lambda m: _extract_quoted_strings_from_bracket(m.group(1)),
            result,
        )
        if result == before:
            break
    return result


def _extract_quoted_strings_from_bracket(content: str) -> str:
    """Extract quoted strings from bracket content, return them comma-separated."""
    quoted = re.findall(r"""['"]([^'"]+)['"]""", content)
    if quoted:
        return ", ".join(f'"{s}"' for s in quoted if s.strip())
    return ""


def _recover_truncated_children_mapping_blocks(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped.startswith("{"):
        return []
    candidates: list[str] = [stripped]
    if stripped.count("[") > stripped.count("]"):
        candidates.append(f"{stripped}]")
    if stripped.count("{") > stripped.count("}"):
        candidates.append(f"{stripped}}}")
    if stripped.count("[") > stripped.count("]") and stripped.count("{") > stripped.count("}"):
        candidates.append(f"{stripped}]}}")
    # Also handle case where list is open but outer brace is closed: {Key: [val1, val2}
    if stripped.endswith("}") and stripped.count("[") > stripped.count("]"):
        candidates.append(f"{stripped[:-1]}]}}")
    # Handle trailing comma before ] or }: ['val1', 'val2', ]} -> ['val1', 'val2']}
    # Pattern: comma followed by close brace and close bracket: '}, ]'
    trailing_comma_fixed = re.sub(r",\s*\}\s*\]", "]", stripped)
    if trailing_comma_fixed != stripped:
        candidates.append(trailing_comma_fixed)
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped

def _recover_children_mapping_from_lines(text: str) -> dict[str, list[str]] | None:
    block = _extract_outer_block(text=text, opener="{", closer="}")
    candidate_text = block if block is not None else text
    sanitized_text = _strip_mapping_comments(candidate_text)
    lines = [line.strip() for line in sanitized_text.splitlines() if line.strip()]
    if not lines:
        return None
    mapping: dict[str, list[str]] = {}
    key: str | None = None
    values: list[str] = []
    in_children = False

    def flush_current_entry() -> None:
        nonlocal key, values, in_children
        if key is not None:
            mapping[key] = values
        key = None
        values = []
        in_children = False

    for line in lines:
        if line in {"{", "}"}:
            if line == "}":
                flush_current_entry()
            continue
        if ":" in line:
            line_key, line_value = line.split(":", 1)
            candidate_key = _extract_quoted_line_value(line_key)
            if candidate_key is not None:
                flush_current_entry()
                key = candidate_key
                if "[" not in line_value:
                    continue
                in_children = True
                value = _extract_quoted_line_value(line_value.split("[", 1)[1])
                if value is not None:
                    values.append(value)
                if "]" in line_value.split("[", 1)[1]:
                    flush_current_entry()
                continue
        if key is None:
            continue
        if not in_children:
            continue
        if line.startswith("]") or line.startswith("}"):
            flush_current_entry()
            continue
        value = _extract_quoted_line_value(line)
        if value is None:
            continue
        values.append(value)
    flush_current_entry()
    if not mapping:
        return None
    if _children_mapping_needs_more_recovery(text=sanitized_text, parsed=mapping):
        return None
    return mapping


def _recover_label_list_from_lines(text: str) -> list[str] | None:
    block = _extract_outer_block(text=text, opener="[", closer="]")
    if block is None:
        block = _recover_truncated_list_block(text)
        if block is None:
            from_segments = _recover_label_list_from_segments(text)
            if from_segments is not None:
                return from_segments
            return _recover_label_list_from_quoted_candidates(text)
    labels: list[str] = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or line in {"[", "]"}:
            continue
        label = _extract_quoted_line_value(line)
        if (
            label is not None
            and "\n" not in block
            and sum(line.count(quote) for quote in ("'", '"')) > 2
        ):
            from_segments = _recover_label_list_from_segments(text)
            if from_segments is not None:
                return from_segments
        if label is None:
            return _recover_label_list_from_segments(text)
        labels.append(label)
    if not labels:
        from_segments = _recover_label_list_from_segments(text)
        if from_segments is not None:
            return from_segments
        return _recover_label_list_from_quoted_candidates(text)
    if len(labels) == 1 and _looks_like_packed_label(labels[0]):
        from_segments = _recover_label_list_from_segments(text)
        if from_segments is not None:
            return from_segments
    return labels


def _recover_truncated_list_block(text: str) -> str | None:
    stripped = text.strip()
    if not stripped.startswith("[") or stripped.endswith("]"):
        return None
    candidate = f"{stripped}]"
    try:
        parsed = ast.literal_eval(candidate)
    except (ValueError, SyntaxError):
        return None
    if (
        not isinstance(parsed, list)
        or not parsed
        or not all(isinstance(item, str) for item in parsed)
    ):
        return None
    return candidate


def _recover_label_list_from_segments(text: str) -> list[str] | None:
    stripped = text.strip()
    if not stripped.startswith("["):
        return None
    body = stripped[1:].strip()
    if not body:
        return None
    segments = [segment.strip() for segment in body.split(",")]
    labels: list[str] = []
    for segment in segments:
        cleaned = segment.strip().strip("[]").strip().strip("'\"")
        if not cleaned:
            continue
        if any(marker in cleaned for marker in ("```", "{", "}", ":", "assistant")):
            return None
        labels.append(cleaned)
    if len(labels) < 2:
        return labels if len(labels) == 1 else None
    return labels


def _recover_label_list_from_quoted_candidates(text: str) -> list[str] | None:
    quoted_items = re.findall(r"""['"]([^'"]+)['"]""", text)
    if len(quoted_items) < 3:
        return None
    labels: list[str] = []
    seen: set[str] = set()
    for item in quoted_items:
        candidate = item.strip()
        if not _looks_like_recoverable_label(candidate):
            continue
        normalized_candidate = candidate.casefold()
        if normalized_candidate in seen:
            continue
        seen.add(normalized_candidate)
        labels.append(candidate)
    if len(labels) < 3:
        return None
    return labels[:5]


def _recover_bare_comma_separated_label_list(text: str) -> list[str] | None:
    stripped = text.strip()
    if not stripped or stripped.startswith("["):
        return None
    if any(marker in stripped for marker in "{}:`"):
        return None
    parts = [part.strip().strip("'\"") for part in stripped.split(",")]
    labels = [part for part in parts if part and _looks_like_recoverable_label(part)]
    if len(labels) < 2:
        return None
    return labels[:5]


def _recover_quoted_label_list_with_comments(text: str) -> list[str] | None:
    if "#" not in text and "**Note:**" not in text and "**note:**" not in text:
        return None
    quoted_items = re.findall(r"""['"]([^'"]+)['"]""", text)
    if len(quoted_items) < 2:
        return None
    labels: list[str] = []
    seen: set[str] = set()
    for item in quoted_items:
        candidate = item.strip()
        if not _looks_like_recoverable_label(candidate):
            continue
        normalized = candidate.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        labels.append(candidate)
    if len(labels) < 2:
        return None
    return labels[:5]


def _recover_single_bare_label(text: str) -> list[str] | None:
    stripped = text.strip()
    if not stripped:
        return None
    if any(marker in stripped for marker in "[]{}:,`\n"):
        return None
    if stripped.casefold().startswith("assistant"):
        return None
    candidate = stripped.strip("'\"").strip()
    if not _looks_like_recoverable_label(candidate):
        return None
    return [candidate]


def _looks_like_recoverable_label(text: str) -> bool:
    if len(text) <= 1:
        return False
    return bool(re.search(r"[A-Za-z]{2,}", text))


def _extract_outer_block(*, text: str, opener: str, closer: str) -> str | None:
    start = text.find(opener)
    end = text.rfind(closer)
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _extract_first_balanced_block(text: str, *, opener: str, closer: str) -> str | None:
    start = text.find(opener)
    if start == -1:
        return None
    depth = 0
    for index in range(start, len(text)):
        if text[index] == opener:
            depth += 1
        elif text[index] == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _looks_like_packed_label(text: str) -> bool:
    return "', '" in text or '", "' in text


def _extract_quoted_line_value(text: str) -> str | None:
    stripped = text.strip().rstrip(",")
    first_quote_positions = [position for position in (stripped.find("'"), stripped.find('"')) if position != -1]
    if not first_quote_positions:
        return None
    start = min(first_quote_positions)
    parsed = _scan_lenient_quoted_string(stripped, start)
    if parsed is not None:
        value, _ = parsed
        cleaned_value = value.strip()
        if (
            len(cleaned_value) >= 2
            and cleaned_value[0] in {"'", '"'}
            and cleaned_value[-1] == cleaned_value[0]
        ):
            cleaned_value = cleaned_value[1:-1].strip()
        return cleaned_value
    return None


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
        "<think>" in lowered
        or "</think>" in lowered
        or not stripped.rstrip().endswith(("]", ")"))
    )


def _looks_like_edge_endpoint(text: str) -> bool:
    if not text:
        return False
    return bool(re.search(r"[A-Za-z0-9]", text))


def _set_generation_temperature(*, model_object: Any, temperature: float) -> None:
    generation_config = getattr(model_object, "generation_config", None)
    if generation_config is None:
        return
    generation_config.temperature = temperature


def _resolve_safety_margin_tokens(context_policy: dict[str, object]) -> int:
    raw_value = context_policy.get("safety_margin_tokens", 64)
    if isinstance(raw_value, int):
        return raw_value
    if isinstance(raw_value, float):
        return int(raw_value)
    if isinstance(raw_value, str):
        return int(raw_value)
    raise ValueError(f"Unsupported safety_margin_tokens value: {raw_value!r}")


def _resolve_generation_timeout_seconds(
    context_policy: dict[str, object],
) -> float | None:
    raw_value = context_policy.get("generation_timeout_seconds")
    if raw_value is None:
        return None
    if isinstance(raw_value, int | float):
        timeout = float(raw_value)
    elif isinstance(raw_value, str):
        timeout = float(raw_value)
    else:
        raise ValueError(f"Unsupported generation_timeout_seconds value: {raw_value!r}")
    if timeout <= 0:
        raise ValueError("generation_timeout_seconds must be positive.")
    return timeout


def _response_hit_generation_limit(
    *,
    completion_ids: Any,
    max_new_tokens: int,
    eos_token_id: int | list[int] | None,
) -> bool:
    completion_length = len(completion_ids)
    if completion_length < max_new_tokens:
        return False
    if completion_length == 0:
        return False
    if eos_token_id is None:
        return True
    eos_ids = _normalize_eos_ids(eos_token_id)
    last_token_id = int(completion_ids[-1])
    return last_token_id not in eos_ids


def _next_max_new_tokens(
    *,
    tokenizer: Any,
    input_token_count: int,
    current_max_new_tokens: int,
    safety_margin_tokens: int,
) -> int:
    max_model_length = _resolve_context_limit(tokenizer)
    if max_model_length is None:
        return current_max_new_tokens * 2
    available_completion_budget = max_model_length - input_token_count - safety_margin_tokens
    if available_completion_budget <= current_max_new_tokens:
        raise ValueError(
            "Model output reached max_new_tokens before EOS; refusing a possibly truncated "
            "response."
        )
    return min(current_max_new_tokens * 2, available_completion_budget)


def _normalize_eos_ids(eos_token_id: int | list[int]) -> set[int]:
    if isinstance(eos_token_id, list):
        return {int(token_id) for token_id in eos_token_id}
    return {int(eos_token_id)}


def _encode_chat_prompt(
    *,
    tokenizer: Any,
    prompt: str,
    model: str,
    device: str,
    disable_thinking: bool,
) -> dict[str, Any]:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        template_kwargs: dict[str, object] = {
            "conversation": messages,
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
            "tokenize": True,
        }
        if disable_thinking:
            if model != _QWEN_CHAT_MODEL:
                raise ValueError(
                    f"Explicit thinking-mode disable is unsupported for model {model}."
                )
            template_kwargs["enable_thinking"] = False
        try:
            encoded = tokenizer.apply_chat_template(**template_kwargs)
        except TypeError as error:
            if disable_thinking:
                raise ValueError(
                    f"Tokenizer for {model} does not support explicit thinking disable."
                ) from error
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
            )
    else:
        if disable_thinking:
            raise ValueError(f"Explicit thinking-mode disable is unsupported for model {model}.")
        encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
    return {key: value.to(device) for key, value in encoded.items()}


def _resolve_pad_token_id(tokenizer: Any) -> int | None:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return int(pad_token_id)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        return int(eos_token_id)
    return None


def _build_runtime_profile(model: str, *, context_limit: int | None) -> RuntimeProfile:
    torch = _torch()
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True

    dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    supports_thinking_toggle = supports_explicit_thinking_disable(model)
    return RuntimeProfile(
        device="cuda",
        dtype=dtype,
        quantization="none",
        supports_thinking_toggle=supports_thinking_toggle,
        context_limit=context_limit,
    )


def _resolve_context_limit(tokenizer: Any) -> int | None:
    max_model_length = getattr(tokenizer, "model_max_length", None)
    if isinstance(max_model_length, int) and 0 < max_model_length < 10**9:
        return max_model_length
    return None


def _dtype_from_profile(torch: Any, dtype_name: str) -> Any:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def _require_cuda(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "hf-transformers local inference requires CUDA; CPU fallback is disabled."
        )


def _generator_kwarg_rejected(error: ValueError) -> bool:
    message = str(error)
    return "model_kwargs" in message and "generator" in message


@lru_cache(maxsize=1)
def _torch() -> Any:
    import torch

    return torch


@lru_cache(maxsize=1)
def _transformers() -> Any:
    import transformers

    return transformers
