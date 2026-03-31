import pytest

from llm_conceptual_modeling.common.hf_transformers import (
    DecodingConfig,
    derive_context_window,
)


class _Tokenizer:
    def __init__(self, *, model_max_length: int = 128) -> None:
        self.model_max_length = model_max_length

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        return list(range(len(text.split())))


def test_decoding_config_rejects_non_zero_temperature() -> None:
    config = DecodingConfig(algorithm="greedy", temperature=0.1)

    with pytest.raises(ValueError, match="temperature=0.0"):
        config.validate()


def test_derive_context_window_rejects_prompt_that_would_overflow() -> None:
    tokenizer = _Tokenizer(model_max_length=12)

    with pytest.raises(ValueError, match="Prompt requires"):
        derive_context_window(
            tokenizer=tokenizer,
            prompt="one two three four five six",
            max_new_tokens=4,
            safety_margin_tokens=4,
        )
