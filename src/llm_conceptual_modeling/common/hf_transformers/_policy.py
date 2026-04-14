from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class RuntimeProfile:
    device: str
    dtype: str
    quantization: str
    supports_thinking_toggle: bool
    context_limit: int | None
