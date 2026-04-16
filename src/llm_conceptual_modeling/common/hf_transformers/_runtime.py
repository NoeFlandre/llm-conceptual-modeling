"""Runtime loading, factory assembly, and hardware resolution for hf-transformers."""

from __future__ import annotations

import gc
import importlib
import importlib.util
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from llm_conceptual_modeling.common.hf_transformers._policy import (
    _MINISTRAL_CHAT_MODEL,
    _QWEN_CHAT_MODEL,
    RuntimeProfile,
    supports_explicit_thinking_disable,
)
from llm_conceptual_modeling.common.hf_transformers._runtime_support import (
    _get_torch,
    _resolve_context_limit,
)

if TYPE_CHECKING:
    pass

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


class HFTransformersRuntimeFactory:
    def __init__(self, *, hf_token: str | None = None) -> None:
        self._hf_token = hf_token
        self._chat_cache: dict[str, tuple[Any, Any, RuntimeProfile]] = {}
        self._embedding_cache: dict[str, tuple[Any, Any, RuntimeProfile]] = {}

    def build_chat_client(
        self,
        *,
        model: str,
        decoding_config: Any,
        max_new_tokens_by_schema: dict[str, int] | None = None,
        context_policy: dict[str, object] | None = None,
        seed: int = 0,
    ) -> Any:
        from llm_conceptual_modeling.common.hf_transformers._client import (
            HFTransformersChatClient,
        )

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

    def build_embedding_client(self, *, model: str) -> Any:
        from llm_conceptual_modeling.common.hf_transformers._client import (
            HFTransformersEmbeddingClient,
        )

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
    model_object.to("cpu")
    del model_object
    gc.collect()
    torch = _get_torch()
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
def _transformers() -> Any:
    import transformers

    return transformers


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
