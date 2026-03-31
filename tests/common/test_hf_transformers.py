import pytest

from llm_conceptual_modeling.common.hf_transformers import (
    DecodingConfig,
    _response_hit_generation_limit,
    derive_context_window,
)


class _Tokenizer:
    def __init__(self, *, model_max_length: int = 128) -> None:
        self.model_max_length = model_max_length

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        return list(range(len(text.split())))


def test_decoding_config_accepts_non_zero_temperature_when_configured() -> None:
    config = DecodingConfig(algorithm="greedy", temperature=0.1)

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


def test_response_hit_generation_limit_detects_possible_output_truncation() -> None:
    assert _response_hit_generation_limit(
        completion_ids=[11, 12, 13, 14],
        max_new_tokens=4,
        eos_token_id=99,
    )


def test_response_hit_generation_limit_accepts_clean_eos_termination() -> None:
    assert not _response_hit_generation_limit(
        completion_ids=[11, 12, 13, 99],
        max_new_tokens=4,
        eos_token_id=99,
    )
