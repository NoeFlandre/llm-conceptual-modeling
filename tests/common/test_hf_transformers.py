import pytest

from llm_conceptual_modeling.common.hf_transformers import (
    DecodingConfig,
    HFTransformersChatClient,
    _response_hit_generation_limit,
    derive_context_window,
)


class _Tokenizer:
    def __init__(self, *, model_max_length: int = 128) -> None:
        self.model_max_length = model_max_length

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        return list(range(len(text.split())))

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        _ = (token_ids, skip_special_tokens)
        return '{"edges": []}'


class _Tensor:
    def __init__(self, values: list[list[int]]) -> None:
        self._values = values
        self.shape = (len(values), len(values[0]) if values else 0)

    def to(self, _device: str):
        return self

    def __getitem__(self, index):
        return self._values[index]


class _ChatTemplateTokenizer(_Tokenizer):
    def apply_chat_template(
        self,
        conversation=None,
        add_generation_prompt: bool = True,
        return_tensors: str = "pt",
        return_dict: bool = True,
        tokenize: bool = True,
        enable_thinking: bool | None = None,
    ):
        _ = (
            conversation,
            add_generation_prompt,
            return_tensors,
            return_dict,
            tokenize,
            enable_thinking,
        )
        return {"input_ids": _Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])}


class _Model:
    def generate(self, **kwargs):
        _ = kwargs
        return [[1, 2, 3, 4, 5, 6, 7, 8, 9, 99]]


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


def test_complete_json_rejects_chat_template_prompt_that_would_overflow_context() -> None:
    tokenizer = _ChatTemplateTokenizer(model_max_length=10)
    client = HFTransformersChatClient(
        model="Qwen/Qwen3.5-9B",
        decoding_config=DecodingConfig(algorithm="greedy"),
        tokenizer=tokenizer,
        model_object=_Model(),
        device="cpu",
        thinking_mode_supported=True,
        max_new_tokens_by_schema={"edge_list": 1},
        context_policy={"safety_margin_tokens": 1},
    )

    with pytest.raises(ValueError, match="Prompt requires"):
        client.complete_json(
            prompt="one two",
            schema_name="edge_list",
            schema={"type": "object"},
        )
