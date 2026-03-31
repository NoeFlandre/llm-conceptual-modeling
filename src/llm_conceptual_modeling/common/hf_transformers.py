from __future__ import annotations

import ast
import json
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from llm_conceptual_modeling.common.structured_output import normalize_structured_response

_QWEN_CHAT_MODEL = "Qwen/Qwen3.5-9B"
_SUPPORTED_CHAT_MODELS = {
    "mistralai/Ministral-3-8B-Instruct-2512",
    _QWEN_CHAT_MODEL,
    "allenai/Olmo-3-7B-Instruct",
}
_SUPPORTED_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"


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
        self._model = model
        self._decoding_config = decoding_config
        self._tokenizer = tokenizer
        self._model_object = model_object
        self._device = device
        self._seed = seed
        self.thinking_mode_supported = thinking_mode_supported
        self._context_policy = context_policy or {}
        self._max_new_tokens_by_schema = max_new_tokens_by_schema or {
            "edge_list": 256,
            "vote_list": 64,
            "label_list": 128,
            "children_by_label": 384,
        }

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
                "temperature": self._decoding_config.temperature,
                "pad_token_id": _resolve_pad_token_id(self._tokenizer),
                "eos_token_id": eos_token_id,
                "use_cache": True,
                **_decoding_kwargs(self._decoding_config),
            }

            torch = _torch()
            torch.manual_seed(self._seed)
            random.seed(self._seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self._seed)
            generator = torch.Generator(device=self._device).manual_seed(self._seed)
            generation_kwargs["generator"] = generator
            with torch.inference_mode():
                generated_ids = self._model_object.generate(**generation_kwargs)
            completion_ids = generated_ids[0][input_length:]
            if _response_hit_generation_limit(
                completion_ids=completion_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
            ):
                next_max_new_tokens = _next_max_new_tokens(
                    tokenizer=self._tokenizer,
                    input_token_count=input_length,
                    current_max_new_tokens=max_new_tokens,
                    safety_margin_tokens=safety_margin_tokens,
                )
                max_new_tokens = next_max_new_tokens
                continue
            text = self._tokenizer.decode(completion_ids, skip_special_tokens=True)
            try:
                parsed_content = _parse_generated_json(text)
            except ValueError as error:
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
                raise error
            return normalize_structured_response(parsed_content, schema_name=schema_name)


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
        hidden_state = outputs.last_hidden_state.detach().cpu().numpy()
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
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, token=self._hf_token)
        profile = _build_runtime_profile(
            model,
            context_limit=_resolve_context_limit(tokenizer),
        )
        model_object = transformers.AutoModelForCausalLM.from_pretrained(
            model,
            token=self._hf_token,
            torch_dtype=_dtype_from_profile(torch, profile.dtype),
            attn_implementation="sdpa",
        )
        model_object.to(profile.device)
        model_object.eval()
        bundle = (tokenizer, model_object, profile)
        self._chat_cache[model] = bundle
        return bundle

    def _load_embedding_bundle(self, model: str) -> tuple[Any, Any, RuntimeProfile]:
        if model != _SUPPORTED_EMBEDDING_MODEL:
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


def _decoding_kwargs(config: DecodingConfig) -> dict[str, object]:
    if config.algorithm == "greedy":
        return {"do_sample": False}
    if config.algorithm == "beam":
        return {"do_sample": False, "num_beams": config.num_beams}
    return {
        "do_sample": False,
        "penalty_alpha": config.penalty_alpha,
        "top_k": config.top_k,
    }


def _parse_generated_json(text: str) -> object:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(stripped)
        except (ValueError, SyntaxError) as error:
            raise ValueError(f"Model did not return valid structured output: {text}") from error


def _resolve_safety_margin_tokens(context_policy: dict[str, object]) -> int:
    raw_value = context_policy.get("safety_margin_tokens", 64)
    if isinstance(raw_value, int):
        return raw_value
    if isinstance(raw_value, float):
        return int(raw_value)
    if isinstance(raw_value, str):
        return int(raw_value)
    raise ValueError(f"Unsupported safety_margin_tokens value: {raw_value!r}")


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
    supports_thinking_toggle = model == _QWEN_CHAT_MODEL
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


@lru_cache(maxsize=1)
def _torch() -> Any:
    import torch

    return torch


@lru_cache(maxsize=1)
def _transformers() -> Any:
    import transformers

    return transformers
