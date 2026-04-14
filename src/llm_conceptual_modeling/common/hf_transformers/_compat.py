from __future__ import annotations

import gc
import importlib
import importlib.util
import random
import time
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

import numpy as np

from llm_conceptual_modeling.common.hf_transformers._parse import (  # noqa: F403
    _looks_retryable_malformed_output,
    _looks_retryable_normalization_failure,
    _parse_generated_json,
    _resolve_malformed_output_retry_limit,
    _should_normalize_exhausted_malformed_edge_list_to_empty,
)
from llm_conceptual_modeling.common.hf_transformers._policy import (  # noqa: F403
    DecodingConfig,
    RuntimeProfile,
    build_default_decoding_grid,  # noqa: F401  # re-exported via __init__
    runtime_generation_overrides,
    should_disable_stateful_guard,  # noqa: F401  # re-exported via __init__
    supports_decoding_config,
    supports_explicit_thinking_disable,
)
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
            f"Prompt requires {required_window} tokens but tokenizer limit is {max_model_length}."
        )
    return required_window


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
            generation_timeout_seconds = _resolve_generation_timeout_seconds(self._context_policy)
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

        transformers = _get_transformers()
        torch = _get_torch()
        _get_require_cuda()(torch)
        _load_chat_tokenizer_fn = _get_load_chat_tokenizer()
        _load_chat_model_fn = _get_load_chat_model()
        tokenizer = _load_chat_tokenizer_fn(
            transformers=transformers,
            model=model,
            hf_token=self._hf_token,
        )
        _build_runtime_profile_fn = _get_build_runtime_profile()
        profile = _build_runtime_profile_fn(
            model,
            context_limit=_resolve_context_limit(tokenizer),
        )
        _dtype_from_profile_fn = _get_dtype_from_profile()
        model_object = _load_chat_model_fn(
            transformers=transformers,
            model=model,
            hf_token=self._hf_token,
            torch_dtype=_dtype_from_profile_fn(torch, profile.dtype),
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

        transformers = _get_transformers()
        torch = _get_torch()
        _get_require_cuda()(torch)
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
    if model == _MINISTRAL_CHAT_MODEL and hasattr(transformers, "Mistral3ForConditionalGeneration"):
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
    if model == "allenai/Olmo-3-7B-Instruct" and _get_flash_attention_available()():
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
    torch = _get_torch()
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


# Module-level lookups so tests can patch via the hf_transformers namespace.
# These use sys.modules to look up the re-exported names in __init__.py,
# allowing monkeypatch.setattr(hf_transformers, "_torch", ...) to take effect.
def _get_torch() -> Any:
    """Look up _torch from the hf_transformers namespace, falling back to local."""
    import sys as _sys

    hf = _sys.modules.get("llm_conceptual_modeling.common.hf_transformers")
    if hf is not None:
        _torch_ref = getattr(hf, "_torch", None)
        if _torch_ref is not None:
            return _torch_ref()
    return _torch()


def _get_transformers() -> Any:
    """Look up _transformers from the hf_transformers namespace, falling back to local."""
    import sys as _sys

    hf = _sys.modules.get("llm_conceptual_modeling.common.hf_transformers")
    if hf is not None:
        _transformers_ref = getattr(hf, "_transformers", None)
        if _transformers_ref is not None:
            return _transformers_ref()
    return _transformers()


def _get_require_cuda() -> Any:
    """Look up _require_cuda from the hf_transformers namespace, falling back to local."""
    import sys as _sys

    hf = _sys.modules.get("llm_conceptual_modeling.common.hf_transformers")
    if hf is not None:
        _require_cuda_ref = getattr(hf, "_require_cuda", None)
        if _require_cuda_ref is not None:
            return _require_cuda_ref
    return _require_cuda


def _get_load_chat_tokenizer() -> Any:
    """Look up _load_chat_tokenizer from the hf_transformers namespace."""
    import sys as _sys

    hf = _sys.modules.get("llm_conceptual_modeling.common.hf_transformers")
    if hf is not None:
        ref = getattr(hf, "_load_chat_tokenizer", None)
        if ref is not None:
            return ref
    return _load_chat_tokenizer


def _get_load_chat_model() -> Any:
    """Look up _load_chat_model from the hf_transformers namespace."""
    import sys as _sys

    hf = _sys.modules.get("llm_conceptual_modeling.common.hf_transformers")
    if hf is not None:
        ref = getattr(hf, "_load_chat_model", None)
        if ref is not None:
            return ref
    return _load_chat_model


def _get_flash_attention_available() -> Any:
    """Look up _flash_attention_available from the hf_transformers namespace."""
    import sys as _sys

    hf = _sys.modules.get("llm_conceptual_modeling.common.hf_transformers")
    if hf is not None:
        ref = getattr(hf, "_flash_attention_available", None)
        if ref is not None:
            return ref
    return _flash_attention_available


def _get_build_runtime_profile() -> Any:
    """Look up _build_runtime_profile from the hf_transformers namespace."""
    import sys as _sys

    hf = _sys.modules.get("llm_conceptual_modeling.common.hf_transformers")
    if hf is not None:
        ref = getattr(hf, "_build_runtime_profile", None)
        if ref is not None:
            return ref
    return _build_runtime_profile


def _get_dtype_from_profile() -> Any:
    """Look up _dtype_from_profile from the hf_transformers namespace."""
    import sys as _sys

    hf = _sys.modules.get("llm_conceptual_modeling.common.hf_transformers")
    if hf is not None:
        ref = getattr(hf, "_dtype_from_profile", None)
        if ref is not None:
            return ref
    return _dtype_from_profile


def _get_load_qwen_dynamic_cache_type() -> Any:
    """Look up _load_qwen_dynamic_cache_type from the hf_transformers namespace."""
    import sys as _sys

    hf = _sys.modules.get("llm_conceptual_modeling.common.hf_transformers")
    if hf is not None:
        ref = getattr(hf, "_load_qwen_dynamic_cache_type", None)
        if ref is not None:
            return ref
    return _load_qwen_dynamic_cache_type
