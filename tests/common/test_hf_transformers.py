import pytest

import llm_conceptual_modeling.common.hf_transformers as hf_transformers
from llm_conceptual_modeling.common.hf_transformers import (
    DecodingConfig,
    HFTransformersChatClient,
    HFTransformersRuntimeFactory,
    _parse_generated_json,
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
    def __init__(
        self,
        *,
        model_max_length: int = 128,
        decoded_text: str = '{"edges": []}',
    ) -> None:
        super().__init__(model_max_length=model_max_length)
        self._decoded_text = decoded_text

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

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        _ = (token_ids, skip_special_tokens)
        return self._decoded_text


class _Model:
    def __init__(self) -> None:
        self.generation_config = type("GenerationConfig", (), {})()

    def generate(self, **kwargs):
        _ = kwargs
        return [[1, 2, 3, 4, 5, 6, 7, 8, 9, 99]]

    def to(self, _device: str):
        return self

    def eval(self):
        return None


class _RejectingGeneratorModel(_Model):
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs):
        self.calls.append(dict(kwargs))
        if "generator" in kwargs:
            raise ValueError(
                "The following `model_kwargs` are not used by the model: ['generator']"
            )
        return [[1, 2, 3, 4, 5, 6, 7, 8, 9, 99]]


class _NoEosAtLimitModel(_Model):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        _ = kwargs
        return [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]


class _CapturingModel(_Model):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs):
        self.calls.append(dict(kwargs))
        return super().generate(**kwargs)


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


def test_complete_json_retries_without_generator_when_model_rejects_it() -> None:
    tokenizer = _ChatTemplateTokenizer(model_max_length=128)
    model = _RejectingGeneratorModel()
    client = HFTransformersChatClient(
        model="Qwen/Qwen3.5-9B",
        decoding_config=DecodingConfig(algorithm="greedy"),
        tokenizer=tokenizer,
        model_object=model,
        device="cpu",
        thinking_mode_supported=True,
        max_new_tokens_by_schema={"edge_list": 4},
        context_policy={"safety_margin_tokens": 1},
        seed=7,
    )

    response = client.complete_json(
        prompt="one two",
        schema_name="edge_list",
        schema={"type": "object"},
    )

    assert response == {"edges": []}
    assert len(model.calls) == 2
    assert "generator" in model.calls[0]
    assert "generator" not in model.calls[1]


def test_complete_json_passes_generation_timeout_as_max_time() -> None:
    tokenizer = _ChatTemplateTokenizer(model_max_length=128)
    model = _CapturingModel()
    client = HFTransformersChatClient(
        model="Qwen/Qwen3.5-9B",
        decoding_config=DecodingConfig(algorithm="greedy"),
        tokenizer=tokenizer,
        model_object=model,
        device="cpu",
        thinking_mode_supported=True,
        max_new_tokens_by_schema={"edge_list": 4},
        context_policy={"safety_margin_tokens": 1, "generation_timeout_seconds": 123},
        seed=7,
    )

    client.complete_json(
        prompt="one two",
        schema_name="edge_list",
        schema={"type": "object"},
    )

    assert model.calls
    assert model.calls[-1]["max_time"] == 123


def test_complete_json_records_generation_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _ChatTemplateTokenizer(model_max_length=128)
    model = _CapturingModel()
    client = HFTransformersChatClient(
        model="Qwen/Qwen3.5-9B",
        decoding_config=DecodingConfig(algorithm="greedy"),
        tokenizer=tokenizer,
        model_object=model,
        device="cpu",
        thinking_mode_supported=True,
        max_new_tokens_by_schema={"edge_list": 4},
        context_policy={"safety_margin_tokens": 1},
        seed=7,
    )
    perf_counter_values = iter([10.0, 10.5])
    monkeypatch.setattr(hf_transformers.time, "perf_counter", lambda: next(perf_counter_values))

    client.complete_json(
        prompt="one two",
        schema_name="edge_list",
        schema={"type": "object"},
    )

    assert client.last_call_metrics == {
        "schema_name": "edge_list",
        "prompt_token_count": 9,
        "completion_token_count": 1,
        "max_new_tokens": 4,
        "duration_seconds": 0.5,
        "tokens_per_second": 2.0,
    }


def test_greedy_decoding_kwargs_do_not_pass_temperature() -> None:
    kwargs = hf_transformers._decoding_kwargs(DecodingConfig(algorithm="greedy"))

    assert kwargs == {"do_sample": False}


def test_beam_decoding_kwargs_do_not_pass_temperature() -> None:
    kwargs = hf_transformers._decoding_kwargs(DecodingConfig(algorithm="beam", num_beams=2))

    assert kwargs == {"do_sample": False, "num_beams": 2}


def test_contrastive_decoding_kwargs_do_not_pass_temperature() -> None:
    kwargs = hf_transformers._decoding_kwargs(
        DecodingConfig(algorithm="contrastive", penalty_alpha=0.2, top_k=4)
    )

    assert kwargs == {
        "custom_generate": "transformers-community/contrastive-search",
        "do_sample": False,
        "penalty_alpha": 0.2,
        "top_k": 4,
        "trust_remote_code": True,
    }


def test_complete_json_passes_trusted_remote_code_for_contrastive_generation() -> None:
    tokenizer = _ChatTemplateTokenizer(model_max_length=128)
    model = _CapturingModel()
    client = HFTransformersChatClient(
        model="allenai/Olmo-3-7B-Instruct",
        decoding_config=DecodingConfig(algorithm="contrastive", penalty_alpha=0.2, top_k=4),
        tokenizer=tokenizer,
        model_object=model,
        device="cpu",
        thinking_mode_supported=False,
        max_new_tokens_by_schema={"edge_list": 4},
        context_policy={"safety_margin_tokens": 1},
        seed=7,
    )

    client.complete_json(
        prompt="one two",
        schema_name="edge_list",
        schema={"type": "object"},
    )

    assert model.calls
    assert model.calls[-1]["custom_generate"] == "transformers-community/contrastive-search"
    assert model.calls[-1]["trust_remote_code"] is True


def test_parse_generated_json_recovers_flat_quoted_edge_list_with_trailing_garbage() -> None:
    actual = _parse_generated_json(
        "['capacity to hire','quality of the sport infrastructure',"
        "'healthy eating choices','quality','cultural norms','unhealthy foods',"
        "'unhealthy','unhealthy reputation','unhealthy foods','unhealthy reputation',"
        "'unhealthy','resil','thinness','resil','thin','Depression','thinness',"
        "'resil','thin','resil','thin','resil','resil,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],",
        schema_name="edge_list",
    )

    assert actual == [
        "capacity to hire",
        "quality of the sport infrastructure",
        "healthy eating choices",
        "quality",
        "cultural norms",
        "unhealthy foods",
        "unhealthy",
        "unhealthy reputation",
        "unhealthy foods",
        "unhealthy reputation",
        "unhealthy",
        "resil",
        "thinness",
        "resil",
        "thin",
        "Depression",
        "thinness",
        "resil",
        "thin",
        "resil",
        "thin",
        "resil",
    ]


def test_parse_generated_json_rejects_corrupt_quoted_edge_list_with_placeholder_tokens() -> None:
    with pytest.raises(ValueError, match="Model did not return valid structured output"):
        _parse_generated_json(
            "['Quality of the sport infrastructure', 'Quality of the sport infrastructure', "
            "'Marketing of healthy foods', 'Marketing of healthy foods', "
            "'Marketing of healthy foods, 'Public support for healthy products', "
            "'Marketing of unhealthy foods', 'Restrictive covenants', "
            "'Marketing ratio of unhealthy to healthy foods']",
            schema_name="edge_list",
        )


def test_chat_client_sets_model_generation_temperature_to_zero() -> None:
    tokenizer = _ChatTemplateTokenizer(model_max_length=128)
    model = _Model()
    model.generation_config.temperature = 0.7
    client = HFTransformersChatClient(
        model="Qwen/Qwen3.5-9B",
        decoding_config=DecodingConfig(algorithm="greedy", temperature=0.0),
        tokenizer=tokenizer,
        model_object=model,
        device="cpu",
        thinking_mode_supported=True,
        max_new_tokens_by_schema={"edge_list": 4},
        context_policy={"safety_margin_tokens": 1},
    )

    _ = client.complete_json(
        prompt="one two",
        schema_name="edge_list",
        schema={"type": "object"},
    )

    assert model.generation_config.temperature == 0.0


def test_parse_generated_json_recovers_newline_vote_list() -> None:
    parsed = hf_transformers._parse_generated_json("Y\nY\nN\nY", schema_name="vote_list")

    assert parsed == ["Y", "Y", "N", "Y"]


def test_parse_generated_json_recovers_unquoted_edge_list() -> None:
    parsed = hf_transformers._parse_generated_json(
        "[(Aesthetics, Mental well-being), (Appetite, Stress)]",
        schema_name="edge_list",
    )

    assert parsed == [
        ("Aesthetics", "Mental well-being"),
        ("Appetite", "Stress"),
    ]


def test_parse_generated_json_ignores_assistant_prefix_for_edge_list() -> None:
    parsed = hf_transformers._parse_generated_json(
        "assistant\n[(Aesthetics, Mental well-being), (Appetite, Stress)]",
        schema_name="edge_list",
    )

    assert parsed == [
        ("Aesthetics", "Mental well-being"),
        ("Appetite", "Stress"),
    ]


def test_parse_generated_json_recovers_valid_tuples_while_skipping_malformed_tuple() -> None:
    parsed = hf_transformers._parse_generated_json(
        "[(Aesthetics, Mental well-being), (no node), (Appetite, Stress)]",
        schema_name="edge_list",
    )

    assert parsed == [
        ("Aesthetics", "Mental well-being"),
        ("Appetite", "Stress"),
    ]


def test_complete_json_accepts_parseable_vote_list_without_eos() -> None:
    tokenizer = _ChatTemplateTokenizer(
        model_max_length=128,
        decoded_text="Y\nN\nY\nN",
    )
    model = _NoEosAtLimitModel()
    client = HFTransformersChatClient(
        model="Qwen/Qwen3.5-9B",
        decoding_config=DecodingConfig(algorithm="greedy", temperature=0.0),
        tokenizer=tokenizer,
        model_object=model,
        device="cpu",
        thinking_mode_supported=True,
        max_new_tokens_by_schema={"vote_list": 4},
        context_policy={"safety_margin_tokens": 1},
    )

    response = client.complete_json(
        prompt="one two",
        schema_name="vote_list",
        schema={"type": "object"},
    )

    assert response == {"votes": ["Y", "N", "Y", "N"]}
    assert model.calls == 1


def test_runtime_factory_uses_mistral3_loader_for_ministral_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(model: str, token: str | None = None, **kwargs):
            calls.append(("auto-tokenizer", model))
            _ = (token, kwargs)
            return _Tokenizer(model_max_length=2048)

    class _MistralCommonBackend:
        @staticmethod
        def from_pretrained(model: str, token: str | None = None, **kwargs):
            calls.append(("mistral-tokenizer", model))
            assert kwargs["trust_remote_code"] is True
            _ = token
            return _Tokenizer(model_max_length=2048)

    class _AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise AssertionError("AutoModelForCausalLM should not be used for Ministral 3.")

    class _Mistral3ForConditionalGeneration:
        @staticmethod
        def from_pretrained(model: str, **kwargs):
            calls.append(("model", model))
            assert kwargs["device_map"] == "auto"
            assert kwargs["quantization_config"] == ("fp8-config", True)
            assert kwargs["trust_remote_code"] is True
            return _Model()

    class _FineGrainedFP8Config:
        def __new__(cls, *, dequantize: bool):
            return ("fp8-config", dequantize)

    class _Transformers:
        AutoTokenizer = _AutoTokenizer
        AutoModelForCausalLM = _AutoModelForCausalLM
        FineGrainedFP8Config = _FineGrainedFP8Config
        MistralCommonBackend = _MistralCommonBackend
        Mistral3ForConditionalGeneration = _Mistral3ForConditionalGeneration

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def is_bf16_supported() -> bool:
            return True

    class _Torch:
        class backends:
            class cuda:
                class matmul:
                    allow_tf32 = False

        cuda = _Cuda()
        bfloat16 = "bfloat16"
        float16 = "float16"
        float32 = "float32"

    monkeypatch.setattr(hf_transformers, "_transformers", lambda: _Transformers)
    monkeypatch.setattr(hf_transformers, "_torch", lambda: _Torch)

    factory = HFTransformersRuntimeFactory(hf_token="hf-token")

    _ = factory.profile_for_chat_model("mistralai/Ministral-3-8B-Instruct-2512")

    assert calls == [
        ("mistral-tokenizer", "mistralai/Ministral-3-8B-Instruct-2512"),
        ("model", "mistralai/Ministral-3-8B-Instruct-2512"),
    ]


def test_runtime_factory_trusts_remote_code_for_qwen_chat_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer_calls: list[dict[str, object]] = []
    model_calls: list[dict[str, object]] = []

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(model: str, **kwargs):
            tokenizer_calls.append({"model": model, **kwargs})
            return _Tokenizer(model_max_length=2048)

    class _AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model: str, **kwargs):
            model_calls.append({"model": model, **kwargs})
            return _Model()

    class _Transformers:
        AutoTokenizer = _AutoTokenizer
        AutoModelForCausalLM = _AutoModelForCausalLM

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def is_bf16_supported() -> bool:
            return True

    class _Torch:
        class backends:
            class cuda:
                class matmul:
                    allow_tf32 = False

        cuda = _Cuda()
        bfloat16 = "bfloat16"
        float16 = "float16"
        float32 = "float32"

    monkeypatch.setattr(hf_transformers, "_transformers", lambda: _Transformers)
    monkeypatch.setattr(hf_transformers, "_torch", lambda: _Torch)

    factory = HFTransformersRuntimeFactory(hf_token="hf-token")

    _ = factory.profile_for_chat_model("Qwen/Qwen3.5-9B")

    assert tokenizer_calls == [
        {
            "model": "Qwen/Qwen3.5-9B",
            "token": "hf-token",
            "trust_remote_code": True,
        }
    ]
    assert model_calls == [
        {
            "model": "Qwen/Qwen3.5-9B",
            "token": "hf-token",
            "dtype": "bfloat16",
            "attn_implementation": "sdpa",
            "trust_remote_code": True,
        }
    ]


def test_runtime_factory_trusts_remote_code_for_ministral_loaders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer_calls: list[dict[str, object]] = []
    model_calls: list[dict[str, object]] = []

    class _MistralCommonBackend:
        @staticmethod
        def from_pretrained(model: str, **kwargs):
            tokenizer_calls.append({"model": model, **kwargs})
            return _Tokenizer(model_max_length=2048)

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise AssertionError("AutoTokenizer should not be used for Ministral 3.")

    class _AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise AssertionError("AutoModelForCausalLM should not be used for Ministral 3.")

    class _Mistral3ForConditionalGeneration:
        @staticmethod
        def from_pretrained(model: str, **kwargs):
            model_calls.append({"model": model, **kwargs})
            return _Model()

    class _FineGrainedFP8Config:
        def __new__(cls, *, dequantize: bool):
            return ("fp8-config", dequantize)

    class _Transformers:
        AutoTokenizer = _AutoTokenizer
        AutoModelForCausalLM = _AutoModelForCausalLM
        FineGrainedFP8Config = _FineGrainedFP8Config
        MistralCommonBackend = _MistralCommonBackend
        Mistral3ForConditionalGeneration = _Mistral3ForConditionalGeneration

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def is_bf16_supported() -> bool:
            return True

    class _Torch:
        class backends:
            class cuda:
                class matmul:
                    allow_tf32 = False

        cuda = _Cuda()
        bfloat16 = "bfloat16"
        float16 = "float16"
        float32 = "float32"

    monkeypatch.setattr(hf_transformers, "_transformers", lambda: _Transformers)
    monkeypatch.setattr(hf_transformers, "_torch", lambda: _Torch)

    factory = HFTransformersRuntimeFactory(hf_token="hf-token")

    _ = factory.profile_for_chat_model("mistralai/Ministral-3-8B-Instruct-2512")

    assert tokenizer_calls == [
        {
            "model": "mistralai/Ministral-3-8B-Instruct-2512",
            "token": "hf-token",
            "trust_remote_code": True,
        }
    ]
    assert model_calls == [
        {
            "model": "mistralai/Ministral-3-8B-Instruct-2512",
            "token": "hf-token",
            "device_map": "auto",
            "dtype": "bfloat16",
            "quantization_config": ("fp8-config", True),
            "trust_remote_code": True,
        }
    ]
