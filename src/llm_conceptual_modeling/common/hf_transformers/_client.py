"""Client classes and helpers for hf-transformers chat and embedding."""

from __future__ import annotations

import random
import time
from typing import Any

import numpy as np

from llm_conceptual_modeling.common.hf_transformers._parse import (
    _looks_retryable_malformed_output,
    _looks_retryable_normalization_failure,
    _parse_generated_json,
    _resolve_malformed_output_retry_limit,
    _should_normalize_exhausted_malformed_edge_list_to_empty,
)
from llm_conceptual_modeling.common.hf_transformers._policy import (
    _QWEN_CHAT_MODEL,
    _TRUSTED_REMOTE_CODE_CHAT_MODELS,
    DecodingConfig,
    runtime_generation_overrides,
    supports_decoding_config,
)
from llm_conceptual_modeling.common.hf_transformers._qwen import (
    _custom_generate_overrides,
    _temporarily_disable_stateful_guard,
)
from llm_conceptual_modeling.common.hf_transformers._runtime_support import (
    _get_torch,
    _resolve_context_limit,
    derive_context_window_from_input_length,
)
from llm_conceptual_modeling.common.structured_output import normalize_structured_response


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
            torch = _get_torch()
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
                    normalized_content: dict[str, object] = {"edges": []}
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

        torch = _get_torch()
        with torch.inference_mode():
            outputs = self._model(**encoded)
        hidden_state = outputs.last_hidden_state.detach().cpu().float().numpy()
        attention_mask = encoded["attention_mask"].detach().cpu().numpy()[..., None]
        masked = hidden_state * attention_mask
        pooled = masked.sum(axis=1) / np.clip(attention_mask.sum(axis=1), 1, None)
        return {text: pooled[index].astype(float).tolist() for index, text in enumerate(texts)}


def _decoding_kwargs(config: DecodingConfig, *, model: str | None = None) -> dict[str, object]:
    if config.algorithm == "greedy":
        return {"do_sample": False}
    if config.algorithm == "beam":
        return {"do_sample": False, "num_beams": config.num_beams}
    kwargs: dict[str, object] = {
        "do_sample": False,
        "penalty_alpha": config.penalty_alpha,
        "top_k": config.top_k,
    }
    if model in _TRUSTED_REMOTE_CODE_CHAT_MODELS:
        kwargs["trust_remote_code"] = True
    return kwargs


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


def _generator_kwarg_rejected(error: ValueError) -> bool:
    message = str(error)
    return "model_kwargs" in message and "generator" in message
