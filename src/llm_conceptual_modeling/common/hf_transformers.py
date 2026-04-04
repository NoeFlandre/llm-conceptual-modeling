from __future__ import annotations

import ast
import json
import random
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

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
        retried_malformed_output = False
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
                **_decoding_kwargs(self._decoding_config),
            }
            if self._model == _QWEN_CHAT_MODEL and self._decoding_config.algorithm == "contrastive":
                generation_kwargs["cache_implementation"] = "dynamic"
            generation_timeout_seconds = _resolve_generation_timeout_seconds(
                self._context_policy
            )
            if generation_timeout_seconds is not None:
                generation_kwargs["max_time"] = generation_timeout_seconds

            torch = _torch()
            torch.manual_seed(self._seed)
            random.seed(self._seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self._seed)
            generator = torch.Generator(device=self._device).manual_seed(self._seed)
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
                    not retried_malformed_output
                    and _looks_retryable_malformed_output(text=text, schema_name=schema_name)
                ):
                    retried_malformed_output = True
                    continue
                raise error
            try:
                normalized_content = normalize_structured_response(
                    parsed_content,
                    schema_name=schema_name,
                )
            except ValueError as error:
                self.last_failed_response_text = text
                if (
                    not retried_malformed_output
                    and _looks_retryable_normalization_failure(
                        parsed_content=parsed_content,
                        schema_name=schema_name,
                        error=error,
                    )
                ):
                    retried_malformed_output = True
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
        attn_implementation="sdpa",
        **remote_code_kwargs,
    )


def _trusted_remote_code_kwargs(model: str) -> dict[str, object]:
    if model in _TRUSTED_REMOTE_CODE_CHAT_MODELS:
        return {"trust_remote_code": True}
    return {}


def _decoding_kwargs(config: DecodingConfig) -> dict[str, object]:
    if config.algorithm == "greedy":
        return {"do_sample": False}
    if config.algorithm == "beam":
        return {"do_sample": False, "num_beams": config.num_beams}
    return {
        "custom_generate": "transformers-community/contrastive-search",
        "do_sample": False,
        "penalty_alpha": config.penalty_alpha,
        "top_k": config.top_k,
        "trust_remote_code": True,
    }


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
    fenced = re.search(r"```(?P<lang>[A-Za-z0-9_-]*)\s*(?P<body>.*?)```", text, flags=re.DOTALL)
    if fenced is not None:
        return fenced.group("body").strip()
    return text


def _normalize_schema_response(parsed: object, *, schema_name: str) -> object:
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
    if schema_name == "edge_list":
        quoted_items = re.findall(r"""['"]([^'"]+)['"]""", stripped)
        if quoted_items and len(quoted_items) % 2 == 1:
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


def _normalize_label_list_payload(parsed: object) -> list[str] | None:
    if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
        return parsed
    if not isinstance(parsed, dict):
        return None
    labels = parsed.get("labels")
    if isinstance(labels, list) and all(isinstance(item, str) for item in labels):
        if len(labels) == 1:
            packed_labels = _split_packed_label_string(labels[0])
            if packed_labels is not None:
                return packed_labels
        return labels
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
    if schema_name == "children_by_label":
        recovered_children = _recover_children_mapping_from_outer_block(text)
        if recovered_children is not None:
            return {"children_by_label": recovered_children}
        recovered_children = _recover_malformed_children_mapping(text)
        if recovered_children is not None:
            return {"children_by_label": recovered_children}
        recovered_children = _recover_inline_children_mapping(text)
        if recovered_children is not None:
            return {"children_by_label": recovered_children}
        recovered_children = _recover_children_mapping_from_lines(text)
        if recovered_children is not None:
            return {"children_by_label": recovered_children}
    if schema_name == "label_list":
        recovered_labels = _recover_label_list_from_lines(text)
        if recovered_labels is not None:
            return recovered_labels
    if schema_name == "edge_list":
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
            raise ValueError(f"Could not parse tuple content: {tuple_matches[0]}")
            return parsed_edges
        quoted_endpoints = _extract_recoverable_edge_endpoints(text)
        if quoted_endpoints is not None:
            return quoted_endpoints
    if schema_name == "vote_list":
        token_matches = re.findall(r"\b[YyNn]\b", text)
        if token_matches:
            return [token.upper() for token in token_matches]
    return None


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
    block = _extract_outer_block(text=text, opener="{", closer="}")
    if block is None:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(block)
        except (json.JSONDecodeError, ValueError, SyntaxError):
            continue
        if isinstance(parsed, dict) and not parsed:
            return {}
        if _looks_like_children_mapping(parsed):
            return parsed
    return None


def _recover_inline_children_mapping(text: str) -> dict[str, list[str]] | None:
    block = _extract_outer_block(text=text, opener="{", closer="}")
    if block is None:
        return None
    mapping: dict[str, list[str]] = {}
    index = 1
    length = len(block)
    while index < length:
        index = _skip_inline_mapping_separators(block, index)
        if index >= length or block[index] == "}":
            break
        key_result = _scan_lenient_quoted_string(block, index)
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


def _scan_lenient_quoted_list(text: str, start_index: int) -> tuple[list[str], int] | None:
    if text[start_index] != "[":
        return None
    index = start_index + 1
    values: list[str] = []
    while index < len(text):
        index = _skip_inline_mapping_separators(text, index)
        if index >= len(text):
            return None
        if text[index] == "]":
            return values, index + 1
        item_result = _scan_lenient_quoted_string(text, index)
        if item_result is None:
            return None
        item, index = item_result
        values.append(item)
        index = _skip_inline_mapping_separators(text, index)
        if index < len(text) and text[index] == ",":
            index += 1
            continue
        if index < len(text) and text[index] == "]":
            return values, index + 1
        return None
    return None


def _scan_lenient_quoted_string(text: str, start_index: int) -> tuple[str, int] | None:
    if start_index >= len(text) or text[start_index] not in {"'", '"'}:
        return None
    opening_quote = text[start_index]
    delimiter_chars = {",", "]", "}", ":"}
    index = start_index + 1
    while index < len(text):
        current_char = text[index]
        if current_char in {"'", '"'}:
            next_index = _skip_inline_mapping_separators(text, index + 1)
            if next_index >= len(text) or text[next_index] in delimiter_chars:
                value = text[start_index + 1 : index].strip()
                return value, index + 1
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


def _recover_children_mapping_from_lines(text: str) -> dict[str, list[str]] | None:
    block = _extract_outer_block(text=text, opener="{", closer="}")
    if block is None:
        return None
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines:
        return None
    key: str | None = None
    values: list[str] = []
    in_children = False
    for line in lines:
        if line in {"{", "}"}:
            continue
        if key is None:
            if ":" not in line:
                return None
            key = _extract_quoted_line_value(line.split(":", 1)[0])
            if key is None or "[" not in line:
                return None
            in_children = True
            continue
        if not in_children:
            continue
        if line.startswith("]") or line.startswith("}"):
            break
        value = _extract_quoted_line_value(line)
        if value is None:
            return None
        values.append(value)
    if key is None or not values:
        return None
    return {key: values}


def _recover_label_list_from_lines(text: str) -> list[str] | None:
    block = _extract_outer_block(text=text, opener="[", closer="]")
    if block is None:
        block = _recover_truncated_list_block(text)
        if block is None:
            return _recover_label_list_from_segments(text)
    labels: list[str] = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or line in {"[", "]"}:
            continue
        label = _extract_quoted_line_value(line)
        if label is None:
            return _recover_label_list_from_segments(text)
        labels.append(label)
    if not labels:
        return _recover_label_list_from_segments(text)
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


def _extract_outer_block(*, text: str, opener: str, closer: str) -> str | None:
    start = text.find(opener)
    end = text.rfind(closer)
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _extract_quoted_line_value(text: str) -> str | None:
    stripped = text.strip().rstrip(",")
    for quote in ("'", '"'):
        start = stripped.find(quote)
        end = stripped.rfind(quote)
        if start != -1 and end > start:
            return stripped[start + 1 : end].strip()
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
