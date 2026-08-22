from collections.abc import Mapping
from types import FunctionType

import pytest

import llm_conceptual_modeling.common.hf_transformers as hf_transformers
import llm_conceptual_modeling.common.hf_transformers._parse as parse_module
import llm_conceptual_modeling.common.hf_transformers._qwen as qwen_module
from llm_conceptual_modeling.common.hf_transformers import (
    DecodingConfig,
    HFTransformersChatClient,
    HFTransformersEmbeddingClient,
    HFTransformersRuntimeFactory,
    _children_mapping,
    _parse_generated_json,
    _response_hit_generation_limit,
    derive_context_window,
)
from llm_conceptual_modeling.common.hf_transformers._parse import (
    _looks_retryable_malformed_output,
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


class _SequentialDecodeTokenizer(_ChatTemplateTokenizer):
    def __init__(self, decoded_texts: list[str], *, model_max_length: int = 128) -> None:
        super().__init__(model_max_length=model_max_length, decoded_text=decoded_texts[0])
        self._decoded_texts = decoded_texts
        self._decode_calls = 0

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        _ = (token_ids, skip_special_tokens)
        index = min(self._decode_calls, len(self._decoded_texts) - 1)
        self._decode_calls += 1
        return self._decoded_texts[index]


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


class _StatefulContrastiveModel(_CapturingModel):
    def __init__(self) -> None:
        super().__init__()
        self._is_stateful = True

    def generate(self, **kwargs):
        assert self._is_stateful is False
        return super().generate(**kwargs)


class _QwenContrastiveModel(_StatefulContrastiveModel):
    def __init__(self) -> None:
        super().__init__()
        self.load_custom_generate_calls: list[dict[str, object]] = []

    def load_custom_generate(self, repo: str, **kwargs):
        self.load_custom_generate_calls.append({"repo": repo, **kwargs})

        def _custom_generate(*args, **inner_kwargs):
            _ = (args, inner_kwargs)
            return [[1, 2, 3, 4, 5, 6, 7, 8, 9, 99]]

        return _custom_generate


class _EmbeddingTokenizer:
    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool,
        truncation: bool,
        return_tensors: str,
    ) -> dict[str, object]:
        _ = (texts, padding, truncation, return_tensors)
        return {"attention_mask": _EmbeddingMask([[1, 1], [1, 0]])}


class _EmbeddingMask:
    def __init__(self, values: list[list[int]]) -> None:
        self._values = values

    def to(self, _device: str):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        import numpy as np

        return np.array(self._values)


class _BFloat16Tensor:
    def __init__(self, values: list[list[list[float]]]) -> None:
        self._values = values
        self.float_called = False

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        self.float_called = True
        return _FloatTensor(self._values)

    def numpy(self):
        raise TypeError("Got unsupported ScalarType BFloat16")


class _FloatTensor:
    def __init__(self, values: list[list[list[float]]]) -> None:
        self._values = values

    def numpy(self):
        import numpy as np

        return np.array(self._values, dtype=float)


class _EmbeddingOutput:
    def __init__(self, tensor: _BFloat16Tensor) -> None:
        self.last_hidden_state = tensor


class _EmbeddingModel:
    def __init__(self, tensor: _BFloat16Tensor) -> None:
        self.tensor = tensor

    def __call__(self, **kwargs):
        _ = kwargs
        return _EmbeddingOutput(self.tensor)


class _TorchStub:
    class _InferenceMode:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            _ = (exc_type, exc, tb)
            return False

    def inference_mode(self):
        return self._InferenceMode()


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
    assert model.calls[0]["max_time"] == 123


def test_complete_json_does_not_pass_generation_timeout_as_max_time_for_contrastive() -> None:
    tokenizer = _ChatTemplateTokenizer(model_max_length=128)
    model = _CapturingModel()
    client = HFTransformersChatClient(
        model="Qwen/Qwen3.5-9B",
        decoding_config=DecodingConfig(algorithm="contrastive", penalty_alpha=0.8, top_k=4),
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
    assert "max_time" not in model.calls[0]


def test_complete_json_retries_once_after_empty_assistant_artifact() -> None:
    tokenizer = _SequentialDecodeTokenizer([" assistant]", '{"edges": []}'])
    model = _CapturingModel()
    client = HFTransformersChatClient(
        model="allenai/Olmo-3-7B-Instruct",
        decoding_config=DecodingConfig(algorithm="greedy"),
        tokenizer=tokenizer,
        model_object=model,
        device="cpu",
        thinking_mode_supported=False,
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


def test_complete_json_retries_multiple_times_for_qwen_contrastive_malformed_output() -> None:
    tokenizer = _SequentialDecodeTokenizer(
        [
            '[\n  ("Prevalene',
            "{ <think>\n\"Okay, let's tackle this problem...",
            '{"edges": []}',
        ]
    )
    model = _CapturingModel()
    client = HFTransformersChatClient(
        model="Qwen/Qwen3.5-9B",
        decoding_config=DecodingConfig(algorithm="contrastive", penalty_alpha=0.8, top_k=4),
        tokenizer=tokenizer,
        model_object=model,
        device="cpu",
        thinking_mode_supported=True,
        max_new_tokens_by_schema={"edge_list": 8},
        context_policy={"safety_margin_tokens": 1},
        seed=7,
    )

    response = client.complete_json(
        prompt="one two",
        schema_name="edge_list",
        schema={"type": "object"},
    )

    assert response == {"edges": []}
    assert len(model.calls) == 3


def test_complete_json_uses_distinct_seeds_for_qwen_contrastive_malformed_output_retries() -> None:
    tokenizer = _SequentialDecodeTokenizer(
        [
            '[\n  ("Prevalene',
            "{ <think>\n\"Okay, let's tackle this problem...",
            '{"edges": []}',
        ]
    )

    class _SeedRecordingModel(_CapturingModel):
        def __init__(self) -> None:
            super().__init__()
            self.generator_seeds: list[int] = []

        def generate(self, **kwargs):
            generator = kwargs.get("generator")
            assert generator is not None
            self.generator_seeds.append(generator.initial_seed())
            return super().generate(**kwargs)

    model = _SeedRecordingModel()
    client = HFTransformersChatClient(
        model="Qwen/Qwen3.5-9B",
        decoding_config=DecodingConfig(algorithm="contrastive", penalty_alpha=0.8, top_k=4),
        tokenizer=tokenizer,
        model_object=model,
        device="cpu",
        thinking_mode_supported=True,
        max_new_tokens_by_schema={"edge_list": 8},
        context_policy={"safety_margin_tokens": 1},
        seed=7,
    )

    response = client.complete_json(
        prompt="one two",
        schema_name="edge_list",
        schema={"type": "object"},
    )

    assert response == {"edges": []}
    assert model.generator_seeds == [7, 8, 9]


def test_complete_json_grows_max_new_tokens_for_qwen_contrastive_malformed_output_retries() -> None:
    tokenizer = _SequentialDecodeTokenizer(
        [
            '[\n  ("Prevalene',
            "{ <think>\n\"Okay, let's tackle this problem...",
            '{"edges": []}',
        ]
    )

    class _MaxTokensRecordingModel(_CapturingModel):
        def __init__(self) -> None:
            super().__init__()
            self.max_new_tokens: list[int] = []

        def generate(self, **kwargs):
            self.max_new_tokens.append(int(kwargs["max_new_tokens"]))
            return super().generate(**kwargs)

    model = _MaxTokensRecordingModel()
    client = HFTransformersChatClient(
        model="Qwen/Qwen3.5-9B",
        decoding_config=DecodingConfig(algorithm="contrastive", penalty_alpha=0.8, top_k=4),
        tokenizer=tokenizer,
        model_object=model,
        device="cpu",
        thinking_mode_supported=True,
        max_new_tokens_by_schema={"edge_list": 8},
        context_policy={"safety_margin_tokens": 1},
        seed=7,
    )

    response = client.complete_json(
        prompt="one two",
        schema_name="edge_list",
        schema={"type": "object"},
    )

    assert response == {"edges": []}
    assert model.max_new_tokens == [8, 16, 32]


def test_complete_json_normalizes_odd_flat_label_list_to_empty_edges() -> None:
    tokenizer = _SequentialDecodeTokenizer(["['A', 'B', 'C', 'D', 'E', 'F', 'G']"])
    model = _CapturingModel()
    client = HFTransformersChatClient(
        model="allenai/Olmo-3-7B-Instruct",
        decoding_config=DecodingConfig(algorithm="greedy"),
        tokenizer=tokenizer,
        model_object=model,
        device="cpu",
        thinking_mode_supported=False,
        max_new_tokens_by_schema={"edge_list": 8},
        context_policy={"safety_margin_tokens": 1},
        seed=7,
    )

    response = client.complete_json(
        prompt="one two",
        schema_name="edge_list",
        schema={"type": "object"},
    )

    assert response == {"edges": []}
    assert len(model.calls) == 1


def test_embedding_client_casts_bfloat16_hidden_state_before_numpy(monkeypatch) -> None:
    monkeypatch.setattr(hf_transformers, "_torch", lambda: _TorchStub())
    hidden_state = _BFloat16Tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 2.0], [999.0, 999.0]],
        ]
    )
    client = HFTransformersEmbeddingClient(
        tokenizer=_EmbeddingTokenizer(),
        model=_EmbeddingModel(hidden_state),
        device="cpu",
    )

    embeddings = client.embed_texts(["first", "second"])

    assert hidden_state.float_called is True
    assert embeddings == {
        "first": [0.5, 0.5],
        "second": [2.0, 2.0],
    }


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
        "do_sample": False,
        "penalty_alpha": 0.2,
        "top_k": 4,
    }


def test_complete_json_disables_qwen_contrastive_stateful_guard() -> None:
    tokenizer = _ChatTemplateTokenizer(model_max_length=128)
    model = _QwenContrastiveModel()
    client = HFTransformersChatClient(
        model="Qwen/Qwen3.5-9B",
        decoding_config=DecodingConfig(algorithm="contrastive", penalty_alpha=0.2, top_k=4),
        tokenizer=tokenizer,
        model_object=model,
        device="cpu",
        thinking_mode_supported=True,
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
    assert callable(model.calls[-1].get("custom_generate"))
    assert model.calls[-1].get("trust_remote_code") is True
    assert model._is_stateful is True
    assert model.load_custom_generate_calls == [
        {"repo": "transformers-community/contrastive-search", "trust_remote_code": True}
    ]


def test_complete_json_includes_dynamic_cache_for_qwen_contrastive() -> None:
    tokenizer = _ChatTemplateTokenizer(model_max_length=128)
    model = _CapturingModel()
    client = HFTransformersChatClient(
        model="Qwen/Qwen3.5-9B",
        decoding_config=DecodingConfig(algorithm="contrastive", penalty_alpha=0.2, top_k=4),
        tokenizer=tokenizer,
        model_object=model,
        device="cpu",
        thinking_mode_supported=True,
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
    assert "cache_implementation" not in model.calls[-1]


def test_complete_json_includes_trust_remote_code_for_mistral_contrastive() -> None:
    tokenizer = _ChatTemplateTokenizer(model_max_length=128)
    model = _CapturingModel()
    client = HFTransformersChatClient(
        model="mistralai/Ministral-3-8B-Instruct-2512",
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
    assert model.calls[-1].get("trust_remote_code") is True
    assert "cache_implementation" not in model.calls[-1]


def test_contrastive_decoding_kwargs_force_trust_remote_code_for_qwen() -> None:
    kwargs = hf_transformers._decoding_kwargs(
        DecodingConfig(algorithm="contrastive", penalty_alpha=0.2, top_k=4),
        model="Qwen/Qwen3.5-9B",
    )
    assert kwargs.get("trust_remote_code") is True


def test_patch_qwen_contrastive_custom_generate_accepts_qwen_dynamic_cache_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DynamicCache:
        pass

    class _QwenDynamicCache:
        pass

    def _custom_generate():
        return None

    _custom_generate.__globals__["DynamicCache"] = _DynamicCache

    monkeypatch.setattr(
        hf_transformers,
        "_load_qwen_dynamic_cache_type",
        lambda: _QwenDynamicCache,
    )

    patched = hf_transformers._patch_qwen_contrastive_custom_generate(_custom_generate)

    assert patched is _custom_generate
    assert patched.__globals__["DynamicCache"] == (_DynamicCache, _QwenDynamicCache)
    assert hasattr(_QwenDynamicCache, "batch_repeat_interleave")
    assert hasattr(_QwenDynamicCache, "crop")
    assert hasattr(_QwenDynamicCache, "batch_select_indices")


def test_patch_qwen_contrastive_custom_generate_leaves_missing_cache_global_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_generate = FunctionType(compile("lambda: None", "<test>", "eval"), {})

    class _QwenDynamicCache:
        pass

    monkeypatch.setattr(
        hf_transformers,
        "_load_qwen_dynamic_cache_type",
        lambda: _QwenDynamicCache,
    )

    patched = hf_transformers._patch_qwen_contrastive_custom_generate(custom_generate)

    assert patched is custom_generate
    assert "DynamicCache" not in custom_generate.__globals__


def test_patch_qwen_contrastive_custom_generate_preserves_existing_cache_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DynamicCache:
        pass

    class _QwenDynamicCache:
        pass

    monkeypatch.setattr(
        hf_transformers,
        "_load_qwen_dynamic_cache_type",
        lambda: _QwenDynamicCache,
    )

    def _already_tuple():
        return None

    _already_tuple.__globals__["DynamicCache"] = (_DynamicCache, _QwenDynamicCache)
    assert hf_transformers._patch_qwen_contrastive_custom_generate(_already_tuple) is _already_tuple

    def _already_same():
        return None

    _already_same.__globals__["DynamicCache"] = _QwenDynamicCache
    assert hf_transformers._patch_qwen_contrastive_custom_generate(_already_same) is _already_same


def test_patch_qwen_contrastive_custom_generate_handles_non_function_callable() -> None:
    class _Callable:
        def __call__(self) -> None:
            return None

    callable_object = _Callable()
    assert (
        hf_transformers._patch_qwen_contrastive_custom_generate(callable_object) is callable_object
    )
    assert hf_transformers._patch_qwen_contrastive_custom_generate(None) is None


def test_get_load_qwen_dynamic_cache_type_falls_back_when_namespace_ref_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(hf_transformers, "_load_qwen_dynamic_cache_type")

    assert qwen_module._get_load_qwen_dynamic_cache_type() is (
        qwen_module._load_qwen_dynamic_cache_type
    )


def test_custom_generate_overrides_requires_qwen_contrastive_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Model:
        def load_custom_generate(self, *_args: object, **_kwargs: object) -> object:
            return object()

    monkeypatch.setattr(
        qwen_module,
        "_patch_qwen_contrastive_custom_generate",
        lambda _custom_generate: "patched",
    )
    model_object = _Model()

    assert (
        qwen_module._custom_generate_overrides(
            model="other/model",
            model_object=model_object,
            decoding_config=DecodingConfig(algorithm="contrastive"),
        )
        == {}
    )
    assert (
        qwen_module._custom_generate_overrides(
            model="Qwen/Qwen3.5-9B",
            model_object=model_object,
            decoding_config=DecodingConfig(algorithm="greedy"),
        )
        == {}
    )
    assert qwen_module._custom_generate_overrides(
        model="Qwen/Qwen3.5-9B",
        model_object=model_object,
        decoding_config=DecodingConfig(algorithm="contrastive"),
    ) == {"custom_generate": "patched"}


def test_patch_dynamic_cache_global_handles_tuple_and_scalar_variants() -> None:
    class _DynamicCache:
        pass

    class _QwenDynamicCache:
        pass

    def custom_generate() -> None:
        return None

    globals_dict: dict[str, object] = {}
    assert (
        qwen_module._patch_dynamic_cache_global(
            custom_generate,
            globals_dict,
            _DynamicCache,
            _QwenDynamicCache,
        )
        is custom_generate
    )
    assert globals_dict["DynamicCache"] == (_DynamicCache, _QwenDynamicCache)

    globals_dict["DynamicCache"] = (_DynamicCache, _QwenDynamicCache)
    qwen_module._patch_dynamic_cache_global(
        custom_generate,
        globals_dict,
        globals_dict["DynamicCache"],
        _QwenDynamicCache,
    )
    assert globals_dict["DynamicCache"] == (_DynamicCache, _QwenDynamicCache)

    globals_dict["DynamicCache"] = (_DynamicCache,)
    qwen_module._patch_dynamic_cache_global(
        custom_generate,
        globals_dict,
        globals_dict["DynamicCache"],
        _QwenDynamicCache,
    )
    assert globals_dict["DynamicCache"] == (_DynamicCache, _QwenDynamicCache)


def test_load_qwen_dynamic_cache_type_reads_the_expected_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _QwenDynamicCache:
        pass

    class _Module:
        Qwen3_5DynamicCache = _QwenDynamicCache

    imported_names: list[str] = []

    def import_module(name: str) -> object:
        imported_names.append(name)
        return _Module()

    monkeypatch.setattr(qwen_module.importlib, "import_module", import_module)

    assert qwen_module._load_qwen_dynamic_cache_type() is _QwenDynamicCache
    assert imported_names == ["transformers.models.qwen3_5.modeling_qwen3_5"]


def test_load_qwen_dynamic_cache_type_returns_none_without_expected_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qwen_module.importlib, "import_module", lambda _name: object())

    assert qwen_module._load_qwen_dynamic_cache_type() is None


def test_ensure_qwen_dynamic_cache_compatibility_preserves_and_adds_methods() -> None:
    batch_repeat_sentinel = object()
    crop_sentinel = object()
    batch_select_sentinel = object()

    class _Existing:
        batch_repeat_interleave = batch_repeat_sentinel
        crop = crop_sentinel
        batch_select_indices = batch_select_sentinel

    qwen_module._ensure_qwen_dynamic_cache_compatibility(_Existing)
    assert _Existing.batch_repeat_interleave is batch_repeat_sentinel
    assert _Existing.crop is crop_sentinel
    assert _Existing.batch_select_indices is batch_select_sentinel

    class _Missing:
        pass

    qwen_module._ensure_qwen_dynamic_cache_compatibility(_Missing)
    assert _Missing.batch_repeat_interleave is qwen_module._qwen_cache_batch_repeat_interleave
    assert _Missing.crop is qwen_module._qwen_cache_crop
    assert _Missing.batch_select_indices is qwen_module._qwen_cache_batch_select_indices


def test_qwen_cache_batch_helpers_transform_all_list_state_attributes() -> None:
    class _Tensor:
        device = "cpu"

        def repeat_interleave(self, repeats: int, *, dim: int) -> tuple[str, int, int]:
            return ("repeated", repeats, dim)

        def index_select(self, dim: int, indices: object) -> tuple[str, int, object]:
            return ("selected", dim, indices)

    class _Indices:
        def to(self, device: object) -> tuple[str, object]:
            return ("indices", device)

    cache = type("Cache", (), {})()
    cache.key_cache = [_Tensor(), None]
    cache.value_cache = [_Tensor()]
    cache.conv_states = "not a list"
    cache.recurrent_states = [None, _Tensor()]

    hf_transformers._qwen_cache_batch_repeat_interleave(cache, 2)
    assert cache.key_cache == [("repeated", 2, 0), None]
    assert cache.value_cache == [("repeated", 2, 0)]
    assert cache.recurrent_states == [None, ("repeated", 2, 0)]

    select_cache = type("Cache", (), {})()
    select_cache.key_cache = [_Tensor(), None]
    select_cache.value_cache = [_Tensor()]
    select_cache.conv_states = "not a list"
    select_cache.recurrent_states = [None, _Tensor()]
    hf_transformers._qwen_cache_batch_select_indices(select_cache, _Indices())
    assert select_cache.key_cache == [("selected", 0, ("indices", "cpu")), None]
    assert select_cache.value_cache == [("selected", 0, ("indices", "cpu"))]
    assert select_cache.recurrent_states == [None, ("selected", 0, ("indices", "cpu"))]


def test_qwen_cache_crop_updates_key_and_value_caches() -> None:
    class _CacheTensor:
        def __init__(self, name: str) -> None:
            self.name = name
            self.slices: list[object] = []

        def __getitem__(self, key: object) -> object:
            self.slices.append(key)
            return f"cropped-{self.name}"

    class _Cache:
        def get_seq_length(self) -> int:
            return 10

    key_tensor = _CacheTensor("key")
    value_tensor = _CacheTensor("value")
    cache = _Cache()
    cache.key_cache = [key_tensor, None]
    cache.value_cache = [value_tensor, None]

    hf_transformers._qwen_cache_crop(cache, -3)

    assert cache.key_cache == ["cropped-key", None]
    assert cache.value_cache == ["cropped-value", None]
    assert key_tensor.slices == [(..., slice(None, 7), slice(None, None))]
    assert value_tensor.slices == [(..., slice(None, 7), slice(None, None))]


def test_qwen_cache_crop_leaves_short_cache_and_non_list_attributes_unchanged() -> None:
    class _Cache:
        key_cache = "not a list"
        value_cache = None

        def get_seq_length(self) -> int:
            return 3

    cache = _Cache()

    hf_transformers._qwen_cache_crop(cache, 3)

    assert cache.key_cache == "not a list"
    assert cache.value_cache is None


def test_qwen_cache_crop_skips_equal_length_and_missing_cache_attributes() -> None:
    class _Tensor:
        def __init__(self) -> None:
            self.slices: list[object] = []

        def __getitem__(self, key: object) -> object:
            self.slices.append(key)
            return self

    class _ExactLength:
        def get_seq_length(self) -> int:
            return 3

    exact = _ExactLength()
    exact.key_cache = [_Tensor()]
    exact.value_cache = [_Tensor()]
    qwen_module._qwen_cache_crop(exact, 3)
    assert exact.key_cache[0].slices == []
    assert exact.value_cache[0].slices == []

    class _MissingCaches:
        def get_seq_length(self) -> int:
            return 10

    qwen_module._qwen_cache_crop(_MissingCaches(), 3)


def test_resolve_qwen_crop_length_treats_zero_as_an_absolute_length() -> None:
    class _Cache:
        def get_seq_length(self) -> int:
            return 10

    assert qwen_module._resolve_qwen_crop_length(_Cache(), 0) == 0


def test_mutate_qwen_cache_state_lists_handles_optional_attributes() -> None:
    class _Tensor:
        pass

    cache = type("Cache", (), {})()
    cache.key_cache = [_Tensor()]
    cache.conv_states = [_Tensor()]

    qwen_module._mutate_qwen_cache_state_lists(cache, lambda tensor: ("changed", tensor))

    assert cache.key_cache[0][0] == "changed"
    assert cache.conv_states[0][0] == "changed"


@pytest.mark.parametrize(
    ("text", "schema_name", "expected"),
    [
        ("", "edge_list", True),
        ("{ <think>draft</think>", "children_by_label", True),
        ("<think>draft</think>", "label_list", True),
        ('["source", "target"]', "edge_list", False),
        ('["source", "target"]   ', "edge_list", False),
        ("[]", "edge_list", False),
        ('["source"', "edge_list", True),
        ('["source", "target", "dangling"]', "edge_list", True),
        ('{"parent": ["child"]}', "children_by_label", False),
        ('{"parent": ["child"]', "children_by_label", True),
        ('{"parent": ["child"]}   ', "children_by_label", False),
        ("plain text", "label_list", False),
    ],
)
def test_looks_retryable_malformed_output_classifies_shape_failures(
    text: str,
    schema_name: str,
    expected: bool,
) -> None:
    assert _looks_retryable_malformed_output(text=text, schema_name=schema_name) is expected


@pytest.mark.parametrize(
    ("parsed_content", "schema_name", "error", "expected"),
    [
        (["source", "target"], "edge_list", ValueError("even number of items"), True),
        (["source", 1], "edge_list", ValueError("even number of items"), False),
        (["source"], "label_list", ValueError("even number of items"), False),
        (["source"], "edge_list", ValueError("other error"), False),
    ],
)
def test_looks_retryable_normalization_failure_classifies_edge_shape_errors(
    parsed_content: object,
    schema_name: str,
    error: ValueError,
    expected: bool,
) -> None:
    assert (
        hf_transformers._looks_retryable_normalization_failure(
            parsed_content=parsed_content,
            schema_name=schema_name,
            error=error,
        )
        is expected
    )


def test_retryable_shape_helpers_preserve_trailing_whitespace_semantics() -> None:
    assert not parse_module._looks_retryable_edge_list_output('["source", "target"]   ')
    assert not parse_module._looks_retryable_children_mapping_output('{"parent": ["child"]}   ')


@pytest.mark.parametrize(
    ("model", "algorithm", "schema_name", "retries", "limit", "text", "expected"),
    [
        (
            "Qwen/Qwen3.5-9B",
            "contrastive",
            "edge_list",
            3,
            3,
            '["source"',
            True,
        ),
        (
            "other/model",
            "contrastive",
            "edge_list",
            3,
            3,
            '["source"',
            False,
        ),
        (
            "Qwen/Qwen3.5-9B",
            "greedy",
            "edge_list",
            3,
            3,
            '["source"',
            False,
        ),
        (
            "Qwen/Qwen3.5-9B",
            "contrastive",
            "label_list",
            3,
            3,
            '["source"',
            False,
        ),
        (
            "Qwen/Qwen3.5-9B",
            "contrastive",
            "edge_list",
            2,
            3,
            '["source"',
            False,
        ),
        (
            "Qwen/Qwen3.5-9B",
            "contrastive",
            "edge_list",
            3,
            3,
            '["source", "target"]',
            False,
        ),
    ],
)
def test_should_normalize_exhausted_malformed_edge_list_to_empty_checks_policy(
    model: str,
    algorithm: str,
    schema_name: str,
    retries: int,
    limit: int,
    text: str,
    expected: bool,
) -> None:
    config = DecodingConfig(algorithm=algorithm)

    assert (
        hf_transformers._should_normalize_exhausted_malformed_edge_list_to_empty(
            model=model,
            decoding_config=config,
            schema_name=schema_name,
            malformed_output_retries=retries,
            malformed_output_retry_limit=limit,
            text=text,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("model", "algorithm", "schema_name", "expected"),
    [
        ("Qwen/Qwen3.5-9B", "contrastive", "edge_list", 3),
        ("other/model", "contrastive", "edge_list", 1),
        ("Qwen/Qwen3.5-9B", "greedy", "edge_list", 1),
        ("Qwen/Qwen3.5-9B", "contrastive", "label_list", 1),
        ("Qwen/Qwen3.5-9B", "contrastive", "children_by_label", 3),
        ("Qwen/Qwen3.5-9B", "contrastive", "other", 1),
    ],
)
def test_resolve_malformed_output_retry_limit_requires_all_policy_conditions(
    model: str,
    algorithm: str,
    schema_name: str,
    expected: int,
) -> None:
    assert (
        hf_transformers._resolve_malformed_output_retry_limit(
            model=model,
            decoding_config=DecodingConfig(algorithm=algorithm),
            schema_name=schema_name,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("assistant\nvalue", "value"),
        ("assistant: value", "value"),
        ("ASSISTANT:\nvalue", "value"),
        ("assistant???", ""),
        ("assistant content", "content"),
        ("assistantAnswer", "assistantAnswer"),
    ],
)
def test_strip_assistant_prefix_handles_supported_prefixes_and_noise(
    text: str,
    expected: str,
) -> None:
    assert hf_transformers._strip_assistant_prefix(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("plain text", "plain text"),
        ('```json\n{"A": 1}\n```', '{"A": 1}'),
        (
            '```json\n{"A```B": ["Child"]}\n```',
            '{"A```B": ["Child"]}',
        ),
        (
            'prefix\n```json\n{"A": "x```B"}\n```',
            '{"A": "x```B"}',
        ),
        (
            '```json{"A": "x```B"}```',
            '{"A": "x```B"}',
        ),
    ],
)
def test_strip_code_fence_handles_plain_and_embedded_fences(
    text: str,
    expected: str,
) -> None:
    assert hf_transformers._strip_code_fence(text) == expected


def test_empty_children_response_recognizes_exact_sentinel_only() -> None:
    assert parse_module._is_empty_children_response('"""json"""')
    assert not parse_module._is_empty_children_response('"""JSON"""')


def test_children_recovery_candidates_prioritize_artifacts_and_fix_trailing_comma() -> None:
    artifact_text = '{"A": ["B"]} <think>noise</think>'
    candidates = parse_module._children_recovery_candidates(artifact_text)
    assert candidates[:3] == [
        '{"A": ["B"]} ',
        '{"A": ["B"]}',
        artifact_text,
    ]

    comma_text = "{Discipline: ['Patience', 'Consistenc', } ]"
    assert "{Discipline: ['Patience', 'Consistenc']" in (
        parse_module._children_recovery_candidates(comma_text)
    )


def test_candidate_helpers_insert_and_append_only_new_values() -> None:
    candidates = ["first", "last"]
    parse_module._insert_unique_candidate(candidates, "middle", 1)
    parse_module._insert_unique_candidate(candidates, "first", 1)
    assert candidates == ["first", "middle", "last"]

    parse_module._append_unique_candidate(candidates, "new")
    parse_module._append_unique_candidate(candidates, "middle")
    assert candidates == ["first", "middle", "last", "new"]


@pytest.mark.parametrize(
    "text",
    [
        '{{"Parent": ["Child"]}',
        '{"Parent": ["Child"]}}',
    ],
)
def test_recover_children_candidate_skips_double_quote_fast_path_for_extra_braces(
    monkeypatch: pytest.MonkeyPatch,
    text: str,
) -> None:
    sentinel = {"sentinel": ["value"]}
    monkeypatch.setattr(
        parse_module,
        "_recover_fenced_python_children_mapping",
        lambda _text: None,
    )
    monkeypatch.setattr(
        parse_module,
        "_recover_double_quoted_children_values",
        lambda _text: sentinel,
    )
    monkeypatch.setattr(
        parse_module,
        "_recover_standard_children_candidate",
        lambda _text: None,
    )

    assert parse_module._recover_children_candidate(text) is None


def test_recover_children_candidate_uses_double_quote_fast_path_for_one_sided_brace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = {"sentinel": ["value"]}
    monkeypatch.setattr(
        parse_module,
        "_recover_fenced_python_children_mapping",
        lambda _text: None,
    )
    monkeypatch.setattr(
        parse_module,
        "_recover_double_quoted_children_values",
        lambda _text: sentinel,
    )
    monkeypatch.setattr(
        parse_module,
        "_recover_standard_children_candidate",
        lambda _text: None,
    )

    assert parse_module._recover_children_candidate('{"Parent": ["Child"]}') == sentinel


@pytest.mark.parametrize(
    ("tuple_text", "expected"),
    [
        ("'Source', \"Target\"", ("Source", "Target")),
        ("'XSource', \"TargetX\"", ("XSource", "TargetX")),
        ("Source, Target, Extra", ("Source", "Target, Extra")),
        ("'', 'Target'", None),
    ],
)
def test_parse_tuple_edge_pair_handles_quotes_delimiters_and_empty_endpoints(
    tuple_text: str,
    expected: tuple[str, str] | None,
) -> None:
    assert parse_module._parse_tuple_edge_pair(tuple_text) == expected


def test_recover_vote_list_response_accepts_both_token_cases() -> None:
    assert parse_module._recover_vote_list_response("y n Y N") == ["Y", "N", "Y", "N"]


def test_normalize_schema_response_wraps_children_mapping_and_recovers_labels() -> None:
    assert hf_transformers._normalize_schema_response(
        {"Parent": ["Child"]},
        schema_name="children_by_label",
    ) == {"children_by_label": {"Parent": ["Child"]}}
    assert hf_transformers._normalize_schema_response(
        "'First', 'Second'",
        schema_name="label_list",
    ) == ["First", "Second"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"Parent": ["Child", "Second"]}', {"Parent": ["Child", "Second"]}),
        ('{"Parent": []}', None),
        ("plain text", None),
    ],
)
def test_recover_single_key_children_values(
    text: str,
    expected: dict[str, list[str]] | None,
) -> None:
    assert hf_transformers._recover_single_key_children_values(text) == expected


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"Parent": ["Child"]}, True),
        ({}, False),
        ({"children_by_label": {"Parent": ["Child"]}}, False),
        ({"children_by_label": []}, False),
        ({1: ["Child"]}, False),
        ({"Parent": "Child"}, False),
        ({"Parent": [1]}, False),
    ],
)
def test_looks_like_children_mapping_validates_mapping_shape(
    payload: object,
    expected: bool,
) -> None:
    assert hf_transformers._looks_like_children_mapping(payload) is expected


def test_strip_mapping_comments_preserves_hashes_inside_quoted_values() -> None:
    text = '{"Parent": ["Child # keep"]} # discard this note\n'

    assert hf_transformers._strip_mapping_comments(text) == ('{"Parent": ["Child # keep"]} \n')


def test_recover_truncated_children_mapping_blocks_adds_missing_delimiters() -> None:
    candidates = hf_transformers._recover_truncated_children_mapping_blocks('{"Parent": ["Child"')

    assert candidates[0] == '{"Parent": ["Child"'
    assert '{"Parent": ["Child"]' in candidates
    assert '{"Parent": ["Child"}' in candidates
    assert '{"Parent": ["Child"]}' in candidates
    assert '{"Parent": ["Child"]' in hf_transformers._recover_truncated_children_mapping_blocks(
        '{"Parent": ["Child", }]'
    )


def test_recover_children_mapping_from_lines_handles_inline_and_multiline_values() -> None:
    text = """{
  "Parent": ["First",
    "Second"
  ],
  "Other": [
    "Third"
  ]
}"""

    assert hf_transformers._recover_children_mapping_from_lines(text) == {
        "Parent": ["First", "Second"],
        "Other": ["Third"],
    }


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"Parent": ["Child"]', {"Parent": ["Child"]}),
        ("plain text", None),
        ('{"Parent": "Child"', None),
    ],
)
def test_recover_malformed_children_mapping_handles_truncated_and_invalid_shapes(
    text: str,
    expected: dict[str, list[str]] | None,
) -> None:
    assert hf_transformers._recover_malformed_children_mapping(text) == expected


@pytest.mark.parametrize(
    ("parsed", "expected"),
    [
        (None, None),
        ({}, {}),
        ({"Parent": ["Child"]}, {"Parent": ["Child"]}),
        ([], None),
    ],
)
def test_normalize_children_mapping_candidate_accepts_only_mapping_shapes(
    parsed: object | None,
    expected: dict[str, list[str]] | None,
) -> None:
    assert _children_mapping._normalize_children_mapping_candidate(parsed) == expected


def test_strip_mapping_comments_removes_hash_comment_at_end_of_text() -> None:
    assert hf_transformers._strip_mapping_comments('{"Parent": ["Child"]} # note') == (
        '{"Parent": ["Child"]} '
    )


def test_strip_mapping_comments_removes_inline_and_parenthetical_notes() -> None:
    assert (
        hf_transformers._strip_mapping_comments('{"Parent": ["Child"]} // discard')
        == '{"Parent": ["Child"]} '
    )
    assert (
        hf_transformers._strip_mapping_comments('{"Parent": ["Child"]} (note: discard)')
        == '{"Parent": ["Child"]} '
    )
    assert (
        hf_transformers._strip_mapping_comments('{\n(irrelevant),\n"Parent": ["Child"]\n}')
        == '{\n\n"Parent": ["Child"]\n}'
    )
    for note in ("NOTE", "WARNING", "Caveat", "Correction", "Actually"):
        assert (
            hf_transformers._strip_mapping_comments(f'{{"Parent": ["Child"]}} ({note}: discard)')
            == '{"Parent": ["Child"]} '
        )
    assert (
        hf_transformers._strip_mapping_comments(
            '{"Parent": ["Child"]} (note with context: discard)'
        )
        == '{"Parent": ["Child"]} '
    )
    assert (
        hf_transformers._strip_mapping_comments('{"Parent": ["Child"]} (noteX: keep)')
        == '{"Parent": ["Child"]} (noteX: keep)'
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("text", None),
        ('["Child"', None),
    ],
)
def test_scan_lenient_quoted_list_rejects_wrong_and_unterminated_inputs(
    text: str,
    expected: tuple[list[str], int] | None,
) -> None:
    assert hf_transformers._scan_lenient_quoted_list(text, 0) == expected


def test_scan_lenient_quoted_list_rejects_empty_list_after_separators() -> None:
    assert hf_transformers._scan_lenient_quoted_list("[ ", 0) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("text", -1),
        ("[[nested]", -1),
        ("[[]]", 4),
    ],
)
def test_skip_nested_bracketed_value_handles_invalid_and_nested_inputs(
    text: str,
    expected: int,
) -> None:
    assert hf_transformers._skip_nested_bracketed_value(text, 0) == expected
    assert hf_transformers._skip_nested_bracketed_value("[][", 2) == -1


def test_scan_nested_list_position_rejects_unclosed_nested_values() -> None:
    assert _children_mapping._scan_nested_list_position("[[nested]", 0, []) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("bare text", None),
        ("'", None),
        ("}: value", None),
    ],
)
def test_scan_unquoted_list_item_handles_missing_delimiters_and_empty_values(
    text: str,
    expected: tuple[str, int] | None,
) -> None:
    assert hf_transformers._scan_unquoted_list_item(text, 0) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("no colon", None),
        (": value", None),
        ("}: value", None),
        ("Parent: value", ("Parent", 6)),
    ],
)
def test_scan_unquoted_mapping_key_handles_invalid_and_valid_inputs(
    text: str,
    expected: tuple[str, int] | None,
) -> None:
    assert hf_transformers._scan_unquoted_mapping_key(text, 0) == expected
    assert hf_transformers._scan_unquoted_mapping_key(text, None) is None


def test_scan_lenient_mapping_key_falls_back_to_unquoted_keys() -> None:
    assert hf_transformers._scan_lenient_mapping_key("Parent: value", 0) == (
        "Parent",
        6,
    )
    assert hf_transformers._scan_lenient_mapping_key('"Parent": value', 0) == (
        "Parent",
        8,
    )
    assert hf_transformers._scan_lenient_mapping_key('"Parent: Label": value', 0) == (
        "Parent: Label",
        15,
    )


@pytest.mark.parametrize(
    "text",
    [
        "plain text",
        '{"Parent": ["Child"]}',
        "{Parent: }",
    ],
)
def test_recover_unquoted_key_comma_separated_rejects_non_matching_shapes(text: str) -> None:
    assert hf_transformers._recover_unquoted_key_comma_separated(text) is None


def test_recover_unquoted_key_comma_separated_rejects_mapping_without_colon() -> None:
    assert hf_transformers._recover_unquoted_key_comma_separated("{Parent value}") is None


@pytest.mark.parametrize(
    ("text", "parsed", "expected"),
    [
        ("plain text", {"Parent": ["Child"]}, False),
        ('{"Parent": ["Child"]}', {"Parent": ["Child"]}, False),
        ('{"Parent": ["\\"dangling\\""]}', {"Parent": ['"dangling"']}, True),
        (
            '{"Parent": ["Child"], "Other": ["Value"]}',
            {"Parent": ["Child"], "Other": ["Value"]},
            False,
        ),
    ],
)
def test_children_mapping_needs_more_recovery_classifies_shapes(
    text: str,
    parsed: dict[str, list[str]],
    expected: bool,
) -> None:
    assert (
        hf_transformers._children_mapping_needs_more_recovery(
            text=text,
            parsed=parsed,
        )
        is expected
    )


def test_recover_inline_children_mapping_returns_none_for_unparseable_text() -> None:
    assert hf_transformers._recover_inline_children_mapping("plain text") is None


def test_inline_children_mapping_blocks_keep_distinct_balanced_and_outer_blocks() -> None:
    blocks = _children_mapping._inline_children_mapping_blocks(
        '{"Parent": ["Child"]} trailing {"Other": ["Value"]}'
    )

    assert len(blocks) == 2


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("plain text", None),
        ('{"Parent": []}', None),
        ('{"Parent": ["Child"]}', {"Parent": ["Child"]}),
        ("{' ': ['Child']}", None),
    ],
)
def test_recover_first_quoted_children_entry_handles_empty_and_valid_values(
    text: str,
    expected: dict[str, list[str]] | None,
) -> None:
    assert hf_transformers._recover_first_quoted_children_entry(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("plain text", None),
        ('{"Parent": []}', None),
        ('{"Parent": ["Child"]}', {"Parent": ["Child"]}),
    ],
)
def test_recover_double_quoted_children_values_handles_empty_and_valid_values(
    text: str,
    expected: dict[str, list[str]] | None,
) -> None:
    assert hf_transformers._recover_double_quoted_children_values(text) == expected


def test_recover_double_quoted_children_values_rejects_blank_key() -> None:
    assert hf_transformers._recover_double_quoted_children_values('" ": ["Child"]') is None


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("no quoted values", ""),
        ("'First', 'Second'", '"First", "Second"'),
    ],
)
def test_extract_quoted_strings_from_bracket_handles_empty_and_quoted_content(
    content: str,
    expected: str,
) -> None:
    assert hf_transformers._extract_quoted_strings_from_bracket(content) == expected


def test_recover_children_mapping_from_lines_returns_none_for_empty_input() -> None:
    assert hf_transformers._recover_children_mapping_from_lines("\n  \n") is None


def test_recover_children_mapping_from_lines_flushes_inline_entry() -> None:
    assert hf_transformers._recover_children_mapping_from_lines('{"Inline": ["Only"]}') == {
        "Inline": ["Only"]
    }


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("", None),
        ('"Parent" ["Child"]', ("Parent", ["Child"])),
        ('"Parent" : []', None),
    ],
)
def test_parse_fenced_children_line_rejects_empty_and_empty_value_lines(
    line: str,
    expected: tuple[str, list[str]] | None,
) -> None:
    assert _children_mapping._parse_fenced_children_line(line) == expected


def test_children_mapping_low_level_shape_and_comment_helpers() -> None:
    for payload in (None, [], "text", 1):
        assert _children_mapping._looks_like_children_mapping(payload) is False

    truncated = '{"Parent": ["Child"]'
    assert _children_mapping._malformed_children_mapping_candidates(truncated) == [
        '{"Parent": ["Child"]}',
        '{"Parent": ["Child"}',
    ]
    assert _children_mapping._malformed_children_mapping_candidates("{Parent: Child") == [
        "{Parent: Child}",
    ]

    balanced = '{"Parent": ["Child"]}'
    assert _children_mapping._children_mapping_blocks(f"prefix {balanced} suffix") == [balanced]
    blocks = _children_mapping._children_mapping_blocks(
        f'prefix {balanced} commentary {{"Other": ["Value"]}}'
    )
    assert blocks == [balanced, f'{balanced} commentary {{"Other": ["Value"]}}']
    assert _children_mapping._children_mapping_candidate_variants(balanced) == (balanced,)
    commented = f"{balanced} # note"
    variants = _children_mapping._children_mapping_candidate_variants(commented)
    assert variants[0] == commented
    assert variants[1] == f"{balanced} "


def test_children_mapping_comment_quote_state_helpers() -> None:
    text = '/* first\nsecond */ {"Parent": ["Child"]}'
    assert _children_mapping._strip_mapping_comments(text) == ' {"Parent": ["Child"]}'
    assert (
        _children_mapping._strip_mapping_hash_comments("'a#b' # drop\n\"c#d\" # drop")
        == "'a#b' \n\"c#d\" "
    )

    assert _children_mapping._is_mapping_quote("'", False, False)
    assert not _children_mapping._is_mapping_quote("'", False, True)
    assert _children_mapping._is_mapping_quote('"', False, False)
    assert not _children_mapping._is_mapping_quote('"', True, False)
    assert _children_mapping._toggle_mapping_quote("'", False, True) == (True, True)
    assert _children_mapping._toggle_mapping_quote('"', True, False) == (True, True)
    assert _children_mapping._is_mapping_hash_comment_start("#", False, False)
    assert not _children_mapping._is_mapping_hash_comment_start("#", True, False)
    assert not _children_mapping._is_mapping_hash_comment_start("#", False, True)
    assert not _children_mapping._is_mapping_hash_comment_start("x", False, False)
    assert _children_mapping._strip_mapping_hash_comments(r"escaped\#keep # drop") == (
        r"escaped\#keep "
    )
    assert _children_mapping._strip_mapping_hash_comments(r"escaped\#") == r"escaped\#"
    assert _children_mapping._strip_mapping_hash_comments("trailing\\") == "trailing\\"
    result: list[str] = []
    assert _children_mapping._consume_mapping_hash_character(
        text="x",
        index=0,
        character="x",
        result=result,
        in_single_quote=False,
        in_double_quote=False,
    ) == (1, False, False)
    assert result == ["x"]
    quoted_result: list[str] = []
    assert _children_mapping._consume_mapping_hash_character(
        text='"',
        index=0,
        character='"',
        result=quoted_result,
        in_single_quote=False,
        in_double_quote=False,
    ) == (1, False, True)
    assert quoted_result == ['"']
    assert _children_mapping._skip_mapping_hash_comment("abc\ndef", 1) == 3
    assert _children_mapping._skip_mapping_hash_comment("abc", 1) == 3


def test_children_mapping_inline_scanner_helpers_cover_boundaries() -> None:
    assert _children_mapping._consume_inline_mapping_token(":", 0, ":") == 1
    assert _children_mapping._consume_inline_mapping_token("", 0, ":") is None
    assert _children_mapping._consume_inline_mapping_token("x", 0, ":") is None
    assert _children_mapping._skip_inline_mapping_comma(",", 0) == 1
    assert _children_mapping._skip_inline_mapping_comma("x", 0) == 0
    assert _children_mapping._skip_inline_mapping_comma("", 0) == 0
    assert _children_mapping._skip_inline_mapping_separators(" \t\nX", 0) == 3
    assert _children_mapping._skip_inline_mapping_separators("\rX", 0) == 1

    mapping = {"Parent": ["old"]}
    _children_mapping._store_inline_children_entry(mapping, "Parent", ["new"])
    assert mapping == {"Parent": ["new"]}
    _children_mapping._store_inline_children_entry(mapping, "Parent", [])
    assert mapping == {"Parent": ["new"]}

    block = '{"Parent": ["Child"]}'
    assert _children_mapping._try_inline_children_parse(block) == {"Parent": ["Child"]}
    assert _children_mapping._inline_children_mapping_finished("abc", 3, 3)
    assert _children_mapping._inline_children_mapping_finished("}", 0, 1)
    assert _children_mapping._inline_children_mapping_blocks(f"prefix {block} suffix") == [block]
    assert _children_mapping._inline_children_mapping_blocks(
        f'prefix {block} commentary {{"Other": ["Value"]}}'
    ) == [block, f'{block} commentary {{"Other": ["Value"]}}']


def test_children_mapping_list_and_unquoted_scanner_helpers_cover_delimiters() -> None:
    assert _children_mapping._scan_lenient_quoted_list('["First", "Second"]', 0) == (
        ["First", "Second"],
        len('["First", "Second"]'),
    )
    assert _children_mapping._scan_lenient_list_position(")", 0, [], lambda *_: None) == (
        [],
        1,
    )
    assert _children_mapping._scan_lenient_list_position(",", 0, [], lambda *_: None) is None
    assert _children_mapping._scan_lenient_list_position('"},', 0, [], lambda *_: None) == (
        [],
        2,
    )
    assert _children_mapping._is_terminal_quoted_list_item('"Child"]', 6)
    assert _children_mapping._is_terminal_quoted_list_item("'Child')", 6)
    assert not _children_mapping._is_terminal_quoted_list_item('"   ', 0)
    assert _children_mapping._is_terminal_quoted_list_item('"}', 0)
    assert _children_mapping._finish_nested_list_position("x?", 1, []) is None
    assert _children_mapping._finish_nested_list_position("x,", 1, []) == 2
    assert _children_mapping._finish_nested_list_position("x]", 1, []) == ([], 2)
    assert _children_mapping._finish_nested_list_position("x)", 1, []) == ([], 2)
    assert _children_mapping._finish_nested_list_position("]", 1, []) is None
    assert _children_mapping._advance_after_lenient_list_item("x,]", 1, []) == 2
    assert _children_mapping._skip_nested_bracketed_value("[]", 2) == -1
    assert _children_mapping._scan_unquoted_list_item("Value}", 0) == ("Value", 5)
    assert _children_mapping._scan_unquoted_mapping_key("prefix Parent: value", len("prefix ")) == (
        "Parent",
        "prefix Parent: value".index(":"),
    )
    assert _children_mapping._scan_unquoted_mapping_key(
        "prefix: Parent:Value:more", len("prefix: ")
    ) == ("Parent", "prefix: Parent:Value:more".index(":", len("prefix: ")))
    assert _children_mapping._scan_unquoted_mapping_key("bad]key: value", 0) is None
    assert _children_mapping._scan_unquoted_mapping_key("bad,key: value", 0) is None
    assert _children_mapping._scan_unquoted_mapping_key("{Parent: Value", 0) == (
        "Parent",
        len("{Parent"),
    )
    assert _children_mapping._scan_unquoted_mapping_key("{ParentX: Value", 0) == (
        "ParentX",
        len("{ParentX"),
    )
    assert _children_mapping._scan_unquoted_mapping_key("'Parent': Value", 0) == (
        "Parent",
        len("'Parent'"),
    )
    assert _children_mapping._scan_unquoted_mapping_key("'XParentX': Value", 0) == (
        "XParentX",
        len("'XParentX'"),
    )


def test_children_mapping_recovery_artifact_helpers_cover_fences_and_quotes() -> None:
    assert _children_mapping._sanitize_children_mapping_text_for_recovery("plain") == "plain"
    assert (
        _children_mapping._sanitize_children_mapping_text_for_recovery('"unfinished')
        == "unfinished"
    )
    assert _children_mapping._sanitize_children_mapping_text_for_recovery('"closed"') == '"closed"'
    assert _children_mapping._sanitize_children_mapping_text_for_recovery("abc\\") == "abc"
    assert _children_mapping._sanitize_children_mapping_text_for_recovery("abc\\\\") == "abc\\"
    assert _children_mapping._sanitize_children_mapping_text_for_recovery("'\" ,") == ('"' + "',")
    assert _children_mapping._sanitize_children_mapping_text_for_recovery(r"\u0041") == "A"
    assert _children_mapping._sanitize_children_mapping_text_for_recovery(r"\u004A") == "J"
    assert (
        _children_mapping._sanitize_children_mapping_text_for_recovery("‘ ’ “ ” ：")
        == "' ' \" \" :"
    )
    assert (
        _children_mapping._strip_fenced_content_artifacts("before\n```JSON\nafter")
        == "before\nafter"
    )
    assert (
        _children_mapping._strip_fenced_content_artifacts("before\n```json\nafter")
        == "before\nafter"
    )
    assert _children_mapping._strip_fenced_content_artifacts("**bold text**") == "bold text"
    assert _children_mapping._strip_fenced_content_artifacts("name** :") == "name:"
    assert _children_mapping._strip_fenced_content_artifacts("name**") == "name"
    fenced = '```json\n{"Parent": ["Child"]}\n```'
    assert _children_mapping._extract_fenced_children_block(fenced) == ('{"Parent": ["Child"]}')
    assert (
        _children_mapping._strip_truncated_fence('```json\n{"Parent": ["Child"]}')
        == '{"Parent": ["Child"]}'
    )
    assert (
        _children_mapping._close_truncated_children_mapping('  {"Parent": ["Child"]}  ')
        == '  {"Parent": ["Child"]}'
    )
    assert (
        _children_mapping._close_truncated_children_mapping('  {"Parent": ["Child"]  ')
        == '  {"Parent": ["Child"]}'
    )
    assert _children_mapping._close_truncated_children_mapping("}]") == "}]"
    assert _children_mapping._recover_fenced_children_fallback('{"Parent": ["Child"]}') == {
        "Parent": ["Child"]
    }
    structured = '{"Parent":\n ["Child"]}'
    assert _children_mapping._recover_structured_fenced_children_mapping(structured) == {
        "Parent": ["Child"]
    }

    assert (
        _children_mapping._normalize_fenced_children_line("  'Parent' ['Child'], ")
        == "'Parent' ['Child']"
    )
    assert _children_mapping._normalize_fenced_children_line("lineX") == "lineX"
    assert (
        _children_mapping._normalize_fenced_children_line("  'Parent' ['Child'] },")
        == "'Parent' ['Child']}"
    )
    assert _children_mapping._normalize_fenced_children_line("}") is None
    assert _children_mapping._normalize_fenced_children_line("},") is None
    assert _children_mapping._match_fenced_children_line("'Parent' ['Child']") == (
        "Parent",
        ["Child"],
    )
    assert _children_mapping._split_fenced_children_values("'First', \"Second\"") == [
        "First",
        "Second",
    ]
    assert _children_mapping._split_fenced_children_values("'XFirstX', \"Second\"") == [
        "XFirstX",
        "Second",
    ]
    assert _children_mapping._match_fenced_children_line("\"'Parent' (Note): ['Child']") == (
        "'Parent'",
        ["Child"],
    )
    assert _children_mapping._match_fenced_children_line("\"'XParent' (Note): ['Child']") == (
        "'XParent'",
        ["Child"],
    )
    assert (
        _children_mapping._extract_fenced_children_block('````json\n{"Parent": ["Child"]}\n````')
        is None
    )
    assert (
        _children_mapping._extract_fenced_children_block('aaa\n{"Parent": ["Child"]}\naaa') is None
    )


def test_structured_fenced_mapping_uses_quote_closed_candidate_before_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block = '{"Parent":\n ["Child"]}'
    expected_candidate = '{"Parent"' + "', " + '"Child"]}'
    seen: list[str] = []

    def recover(candidate: str) -> dict[str, list[str]] | None:
        seen.append(candidate)
        return {"Parent": ["Child"]}

    monkeypatch.setattr(_children_mapping, "_recover_first_quoted_children_entry", recover)
    monkeypatch.setattr(
        _children_mapping,
        "_try_recovered_fenced_children_mapping",
        lambda _candidate: None,
    )

    assert _children_mapping._recover_structured_fenced_children_mapping(block) == {
        "Parent": ["Child"]
    }
    assert seen == [expected_candidate]


def test_children_mapping_unquoted_and_truncated_recovery_helpers() -> None:
    assert not _children_mapping._is_unquoted_children_mapping_candidate("{Parent: Value]")
    assert _children_mapping._split_unquoted_children_mapping("{Parent: Value: Two}") == (
        "Parent",
        "Value: Two",
    )
    assert _children_mapping._split_unquoted_children_mapping("{Parent:Value: Two}") == (
        "Parent",
        "Value: Two",
    )
    assert _children_mapping._split_unquoted_children_mapping("{Parent:ValueX}") == (
        "Parent",
        "ValueX",
    )
    assert _children_mapping._parse_unquoted_children_values("'First', Second, \"Third\"") == [
        "First",
        "Second",
        "Third",
    ]
    assert _children_mapping._parse_unquoted_children_values("'XFirstX'") == ["XFirstX"]
    assert _children_mapping._children_values_need_more_recovery(["[dangling"])
    assert _children_mapping._child_value_needs_more_recovery('"leading')
    assert _children_mapping._child_value_needs_more_recovery("'leading")
    assert _children_mapping._child_value_needs_more_recovery("trailing'")
    assert _children_mapping._child_value_needs_more_recovery('trailing"')
    assert _children_mapping._child_value_needs_more_recovery("contains]")
    assert _children_mapping._children_text_needs_more_recovery("text [bare]")
    assert _children_mapping._remove_nonstring_bracket_patterns("[diet] reminders") == ""
    assert (
        _children_mapping._remove_nonstring_bracket_patterns("[nested [a 'x' y 'z']]") == "[nested "
    )
    assert _children_mapping._remove_nonstring_bracket_patterns("[b[[ ' [cc'b[]") == '" [cc"'
    assert _children_mapping._remove_nonstring_bracket_patterns("[''[]") == ""
    assert (
        _children_mapping._truncated_children_mapping_delimiter_candidates('{"Parent": []}') == []
    )
    assert _children_mapping._truncated_children_mapping_delimiter_candidates(
        '{"Parent": "Child"'
    ) == ['{"Parent": "Child"}']
    assert _children_mapping._deduplicate_text_candidates(["a", "a", "b"]) == [
        "a",
        "b",
    ]


def test_children_mapping_line_recovery_helpers_preserve_state_contracts() -> None:
    def extract_outer_block(**kwargs: object) -> str | None:
        if kwargs.get("opener") != "{" or kwargs.get("closer") != "}":
            return None
        return '{"Parent": ["Child"]}'

    prepared = _children_mapping._prepare_children_mapping_lines("ignored", extract_outer_block)
    assert prepared == ('{"Parent": ["Child"]}', ['{"Parent": ["Child"]}'])

    def extract_value(line: str) -> str | None:
        if '"' not in line:
            return None
        return line.split('"')[1]

    mapping: dict[str, list[str]] = {}
    assert _children_mapping._consume_children_mapping_assignment(
        line='"Parent": ["Child: value"]',
        mapping=mapping,
        key=None,
        values=[],
        in_children=False,
        extract_quoted_line_value=extract_value,
    ) == (None, [], False)
    assert mapping == {"Parent": ["Child: value"]}

    def extract_assignment_value(line: str) -> str | None:
        if line == '"New"':
            return "New"
        if line.startswith('"Value"'):
            return "Value"
        return None

    mapping = {}
    assert _children_mapping._consume_children_mapping_assignment(
        line='"New": ["Value"]',
        mapping=mapping,
        key="Old",
        values=["old"],
        in_children=True,
        extract_quoted_line_value=extract_assignment_value,
    ) == (None, [], False)
    assert mapping == {"Old": ["old"], "New": ["Value"]}

    def extract_fallback_value(line: str) -> str | None:
        if line.endswith('"Value"'):
            return "Value"
        return None

    assert _children_mapping._consume_children_mapping_assignment(
        line='unquoted: "Value"',
        mapping={},
        key="Parent",
        values=[],
        in_children=True,
        extract_quoted_line_value=extract_fallback_value,
    ) == ("Parent", ["Value"], True)

    assert _children_mapping._consume_children_mapping_assignment(
        line="}: value",
        mapping={"Parent": ["Child"]},
        key="Parent",
        values=["Child"],
        in_children=True,
        extract_quoted_line_value=extract_fallback_value,
    ) == (None, [], False)

    assert _children_mapping._consume_children_mapping_line(
        line='unquoted: "Value"',
        mapping={},
        key="Parent",
        values=[],
        in_children=True,
        extract_quoted_line_value=extract_fallback_value,
    ) == ("Parent", ["Value"], True)

    seen_value_parts: list[str] = []

    def record_value_part(line: str) -> str | None:
        seen_value_parts.append(line)
        return "New" if line == '"New"' else None

    assert _children_mapping._consume_children_mapping_assignment(
        line='"New": [["Value"]',
        mapping={},
        key=None,
        values=[],
        in_children=False,
        extract_quoted_line_value=record_value_part,
    ) == (None, [], False)
    assert seen_value_parts == ['"New"', '["Value"]']

    def extract_brace_value(line: str) -> str | None:
        if line == "{":
            return "Unexpected"
        return extract_value(line)

    assert _children_mapping._consume_children_mapping_line(
        line="{",
        mapping={},
        key="Parent",
        values=["Child"],
        in_children=True,
        extract_quoted_line_value=extract_brace_value,
    ) == ("Parent", ["Child"], True)
    closed_mapping: dict[str, list[str]] = {}
    assert _children_mapping._consume_children_mapping_line(
        line="}",
        mapping=closed_mapping,
        key="Parent",
        values=["Child"],
        in_children=True,
        extract_quoted_line_value=extract_value,
    ) == (None, [], False)
    assert closed_mapping == {"Parent": ["Child"]}

    for closing_line in ("]", "}"):
        closed_mapping = {}
        assert _children_mapping._consume_children_mapping_value(
            line=closing_line,
            mapping=closed_mapping,
            key="Parent",
            values=["Child"],
            in_children=True,
            extract_quoted_line_value=extract_value,
        ) == (None, [], False)
        assert closed_mapping == {"Parent": ["Child"]}

    assert _children_mapping._recover_children_mapping_from_lines('{\n"Parent": [\n"Child"\n') == {
        "Parent": ["Child"]
    }

    assert _children_mapping._consume_children_mapping_value(
        line='"Value"',
        mapping={},
        key=None,
        values=[],
        in_children=True,
        extract_quoted_line_value=extract_value,
    ) == (None, [], True)
    assert _children_mapping._consume_children_mapping_value(
        line='"Value"',
        mapping={},
        key="Parent",
        values=[],
        in_children=False,
        extract_quoted_line_value=extract_value,
    ) == ("Parent", [], False)
    assert _children_mapping._flush_children_mapping_line_entry(mapping, "Parent", ["Child"]) == (
        None,
        [],
        False,
    )


def test_recover_children_mapping_initializes_line_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[object, object]] = []

    def capture_state(**kwargs: object) -> tuple[None, object, object]:
        if not observed:
            observed.append((kwargs["values"], kwargs["in_children"]))
        return None, kwargs["values"], kwargs["in_children"]

    monkeypatch.setattr(
        _children_mapping,
        "_consume_children_mapping_line",
        capture_state,
    )

    _children_mapping._recover_children_mapping_from_lines('{"Parent": ["Child"]}')

    assert observed == [([], False)]


def test_load_qwen_dynamic_cache_type_propagates_non_import_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(module_name: str) -> object:
        _ = module_name
        raise RuntimeError("boom")

    monkeypatch.setattr(hf_transformers.importlib, "import_module", boom)

    with pytest.raises(RuntimeError, match="boom"):
        hf_transformers._load_qwen_dynamic_cache_type()


def test_contrastive_decoding_kwargs_force_trust_remote_code_for_mistral() -> None:
    kwargs = hf_transformers._decoding_kwargs(
        DecodingConfig(algorithm="contrastive", penalty_alpha=0.2, top_k=4),
        model="mistralai/Ministral-3-8B-Instruct-2512",
    )
    assert kwargs.get("trust_remote_code") is True


def test_contrastive_decoding_kwargs_do_not_force_cache_implementation() -> None:
    """Transformers already handles contrastive cache setup internally."""
    kwargs = hf_transformers._decoding_kwargs(
        DecodingConfig(algorithm="contrastive", penalty_alpha=0.2, top_k=4),
        model="Qwen/Qwen3.5-9B",
    )
    assert "cache_implementation" not in kwargs


def test_runtime_factory_can_prefetch_models() -> None:
    factory = HFTransformersRuntimeFactory(hf_token="token")
    chat_models_seen: list[str] = []
    embedding_models_seen: list[str] = []
    released_chat_models: list[str] = []
    released_embedding_models: list[str] = []
    profile = hf_transformers.RuntimeProfile(
        device="cuda",
        dtype="bfloat16",
        quantization="none",
        supports_thinking_toggle=False,
        context_limit=None,
    )

    def fake_load_chat_bundle(model: str):
        chat_models_seen.append(model)
        return (_Tokenizer(), _Model(), profile)

    def fake_load_embedding_bundle(model: str):
        embedding_models_seen.append(model)
        return (_EmbeddingTokenizer(), _EmbeddingModel(_BFloat16Tensor([[[1.0]]])), profile)

    factory._load_chat_bundle = fake_load_chat_bundle  # type: ignore[method-assign]
    factory._load_embedding_bundle = fake_load_embedding_bundle  # type: ignore[method-assign]
    factory._release_prefetched_chat_bundle = released_chat_models.append  # type: ignore[method-assign]
    factory._release_prefetched_embedding_bundle = released_embedding_models.append  # type: ignore[method-assign]

    report = factory.prefetch_models(
        chat_models=["Qwen/Qwen3.5-9B", "allenai/Olmo-3-7B-Instruct"],
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
    )

    assert chat_models_seen == ["Qwen/Qwen3.5-9B", "allenai/Olmo-3-7B-Instruct"]
    assert embedding_models_seen == ["Qwen/Qwen3-Embedding-0.6B"]
    assert released_chat_models == ["Qwen/Qwen3.5-9B", "allenai/Olmo-3-7B-Instruct"]
    assert released_embedding_models == ["Qwen/Qwen3-Embedding-0.6B"]
    assert report["chat_models"] == chat_models_seen
    assert report["embedding_model"] == "Qwen/Qwen3-Embedding-0.6B"


def test_runtime_factory_skips_manual_device_move_for_accelerate_dispatched_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DispatchedModel(_Model):
        def __init__(self) -> None:
            super().__init__()
            self.hf_device_map = {"layer": "cuda:0"}
            self.to_calls: list[str] = []

        def to(self, device: str):
            self.to_calls.append(device)
            raise RuntimeError(
                "You can't move a model that has some modules offloaded to cpu or disk."
            )

    dispatched_model = _DispatchedModel()

    monkeypatch.setattr(hf_transformers, "_transformers", lambda: object())
    monkeypatch.setattr(hf_transformers, "_torch", lambda: object())
    monkeypatch.setattr(hf_transformers, "_require_cuda", lambda _torch: None)
    monkeypatch.setattr(
        hf_transformers,
        "_load_chat_tokenizer",
        lambda **_: _Tokenizer(model_max_length=2048),
    )
    monkeypatch.setattr(
        hf_transformers,
        "_build_runtime_profile",
        lambda model, context_limit: hf_transformers.RuntimeProfile(
            device="cuda",
            dtype="bfloat16",
            quantization="none",
            supports_thinking_toggle=False,
            context_limit=context_limit,
        ),
    )
    monkeypatch.setattr(
        hf_transformers,
        "_load_chat_model",
        lambda **_: dispatched_model,
    )
    monkeypatch.setattr(
        hf_transformers,
        "_dtype_from_profile",
        lambda _torch, _dtype: "bfloat16",
    )

    factory = HFTransformersRuntimeFactory(hf_token="token")

    _, model_object, profile = factory._load_chat_bundle("Qwen/Qwen3.5-9B")

    assert model_object is dispatched_model
    assert profile.device == "cuda"
    assert dispatched_model.to_calls == []


def test_release_prefetched_model_object_propagates_cleanup_errors() -> None:
    class _BrokenModel:
        def to(self, device: str):
            _ = device
            raise RuntimeError("cleanup failed")

    with pytest.raises(RuntimeError, match="cleanup failed"):
        hf_transformers._release_prefetched_model_object(_BrokenModel())


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


def test_parse_generated_json_recovers_bare_comma_separated_edge_pair() -> None:
    parsed = hf_transformers._parse_generated_json(
        "A, D",
        schema_name="edge_list",
    )

    assert parsed == [("A", "D")]


def test_parse_generated_json_recovers_unparenthesized_bracketed_edge_pairs() -> None:
    parsed = hf_transformers._parse_generated_json(
        "[Medications, Marketing], [Obesity, Eating disorders]",
        schema_name="edge_list",
    )

    assert parsed == [
        ("Medications", "Marketing"),
        ("Obesity", "Eating disorders"),
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


def test_complete_json_normalizes_exhausted_truncated_single_edge_endpoint_to_empty() -> None:
    tokenizer = _SequentialDecodeTokenizer(
        [
            '[\n  (\n    "Prevalen\n</think>',
            '[\n  (\n    "Prevalen\n</think>',
            '[\n  (\n    "Prevalen\n</think>',
            '[\n  (\n    "Prevalen\n</think>',
        ]
    )
    model = _CapturingModel()
    client = HFTransformersChatClient(
        model="Qwen/Qwen3.5-9B",
        decoding_config=DecodingConfig(algorithm="contrastive", penalty_alpha=0.8, top_k=4),
        tokenizer=tokenizer,
        model_object=model,
        device="cpu",
        thinking_mode_supported=True,
        max_new_tokens_by_schema={"edge_list": 8},
        context_policy={"safety_margin_tokens": 1},
        seed=7,
    )

    response = client.complete_json(
        prompt="one two",
        schema_name="edge_list",
        schema={"type": "object"},
    )

    assert response == {"edges": []}
    assert len(model.calls) == 4


def test_parse_generated_json_recovers_bare_children_by_label_mapping() -> None:
    parsed = hf_transformers._parse_generated_json(
        """{
  "Prevalence of walking trails": [
    "Density of trail networks",
    "Accessibility of trail systems",
    "Maintenance frequency of trails"
  ]
}""",
        schema_name="children_by_label",
    )

    assert parsed == {
        "children_by_label": {
            "Prevalence of walking trails": [
                "Density of trail networks",
                "Accessibility of trail systems",
                "Maintenance frequency of trails",
            ]
        }
    }


def test_parse_generated_json_recovers_fenced_python_children_by_label_mapping() -> None:
    parsed = hf_transformers._parse_generated_json(
        """```python
{
    'Prevalence of walking trails': [
        'Density of walking trails per square kilometer',
        'Length of maintained walking trails (kilometers)',
        'Proportion of urban areas with accessible walking trails'
    ]
}
```""",
        schema_name="children_by_label",
    )

    assert parsed == {
        "children_by_label": {
            "Prevalence of walking trails": [
                "Density of walking trails per square kilometer",
                "Length of maintained walking trails (kilometers)",
                "Proportion of urban areas with accessible walking trails",
            ]
        }
    }


def test_parse_generated_json_recovers_malformed_children_by_label_mapping() -> None:
    malformed_response = (
        "{'Supportive food environment' : "
        "['Nutrient-dense food access', "
        "'Food culture diversity', "
        "'Community-supported agriculture proximity']]"
    )
    parsed = hf_transformers._parse_generated_json(
        malformed_response,
        schema_name="children_by_label",
    )

    assert parsed == {
        "children_by_label": {
            "Supportive food environment": [
                "Nutrient-dense food access",
                "Food culture diversity",
                "Community-supported agriculture proximity",
            ]
        }
    }


def test_parse_generated_json_recovers_multikey_children_mapping_missing_closing_brace() -> None:
    malformed_response = (
        "{'A' : ['Fresh produce access', 'Nutrient-dense supply'], "
        "'B' : ['Healthy eating habits', 'Food variety']"
    )
    parsed = hf_transformers._parse_generated_json(
        malformed_response,
        schema_name="children_by_label",
    )

    assert parsed == {
        "children_by_label": {
            "A": ["Fresh produce access", "Nutrient-dense supply"],
            "B": ["Healthy eating habits", "Food variety"],
        }
    }


def test_parse_generated_json_recovers_single_entry_children_mapping_with_empty_value() -> None:
    parsed = hf_transformers._parse_generated_json(
        """{
  "'Cultural norms aroundthinness'":
}""",
        schema_name="children_by_label",
    )

    assert parsed == {
        "children_by_label": {
            "Cultural norms aroundthinness": [],
        }
    }


def test_parse_generated_json_recovers_children_mapping_with_inner_apostrophe() -> None:
    parsed = hf_transformers._parse_generated_json(
        """```python
{
    'Accessibility and affordability of nutritious food options': [
        'Proximity of grocery stores and farmers' markets to residential areas',
        'Price volatility and stability of essential food commodities',
        'Availability of government subsidies and food assistance programs'
    ]
}
```""",
        schema_name="children_by_label",
    )

    assert parsed == {
        "children_by_label": {
            "Accessibility and affordability of nutritious food options": [
                "Proximity of grocery stores and farmers' markets to residential areas",
                "Price volatility and stability of essential food commodities",
                "Availability of government subsidies and food assistance programs",
            ]
        }
    }


def test_parse_generated_json_recovers_children_mapping_with_inline_fence() -> None:
    parsed = hf_transformers._parse_generated_json(
        """```python{
    'Prevalence of walking trails': [
        'Length of pedestrian pathways per capita (km/person)',
        '% Population engaging in walking for commuting',
        '# Publicly maintained walking trails normalized by area'
    ]
}```""",
        schema_name="children_by_label",
    )

    assert parsed == {
        "children_by_label": {
            "Prevalence of walking trails": [
                "Length of pedestrian pathways per capita (km/person)",
                "% Population engaging in walking for commuting",
                "# Publicly maintained walking trails normalized by area",
            ]
        }
    }


def test_parse_generated_json_recovers_fenced_children_mapping_without_outer_braces() -> None:
    parsed = hf_transformers._parse_generated_json(
        """```python{
"% Population Engaging in Walk-for-Commuting": [
    "% Population Walking >5 km per day (any purpose)",  # Higher/lower frequency of long-duration walking in daily life  # noqa: E501
    "Average Distance Traveled on Footscray by Residents (km/person/annum)"  #
        (quantitative measure of foot travel beyond commuting context),
    "'Car-Freedom Index' (% Households Without Personal Vehicl Access)"  #
        Inverse proxy for reliance/redundancy of non-walk modes
]
```""",  # noqa: E501
        schema_name="children_by_label",
    )

    assert parsed == {
        "children_by_label": {
            "% Population Engaging in Walk-for-Commuting": [
                "% Population Walking >5 km per day (any purpose)",
                "Average Distance Traveled on Footscray by Residents (km/person/annum)",
                "'Car-Freedom Index' (% Households Without Personal Vehicl Access)",
            ]
        }
    }


def test_parse_generated_json_recovers_fenced_children_mapping_without_braces_or_dict_wrapper() -> (
    None
):
    parsed = hf_transformers._parse_generated_json(
        """```python
'Supportive食環境': [
    'Convenience of nutritious food access',
    'Affordabiltiy of wholesomeness in diet',
    'Proximity-to-health-promoters (e.g., farmers markets/grocery stores)',  # Less = distance to fast/sugarcalorie sources increases  # noqa: E501
\t'Harmonized school/univerisity cafeteria policies (e.‚Äögreens by default, sugar tax on desser)t',
    'Urban/rur‚ÅÑÔ∏è food desert vs oasismapping (extent of "green zones")'
]
```""",  # noqa: E501
        schema_name="children_by_label",
    )

    assert parsed == {
        "children_by_label": {
            "Supportive食環境": [
                "Convenience of nutritious food access",
                "Affordabiltiy of wholesomeness in diet",
                "Proximity-to-health-promoters (e.g., farmers markets/grocery stores)",
                "Harmonized school/univerisity cafeteria policies (e.‚Äögreens by default, sugar tax on desser)t",  # noqa: E501
                'Urban/rur‚ÅÑÔ∏è food desert vs oasismapping (extent of "green zones")',
            ]
        }
    }


def test_parse_generated_json_recovers_children_mapping_with_trailing_note_after_empty_dict() -> (
    None
):
    parsed = hf_transformers._parse_generated_json(
        (
            "{}\n\n"
            "(Note: With only one concept in the input---'Eating disorders'---there "
            "are no concepts to pair.)"
        ),
        schema_name="children_by_label",
    )

    assert parsed == {"children_by_label": {}}


def test_parse_generated_json_recovers_inline_children_mapping_with_duplicates() -> None:
    parsed = hf_transformers._parse_generated_json(
        (
            "{'A' : ['Access to fresh produce', 'Nutritional diversity'], "
            "'H' : ['Food budgeting practices', 'Farmers' market presence', "
            "'Nutritional labeling accuracy'], 'A' : [], 'H' : []}"
        ),
        schema_name="children_by_label",
    )

    assert parsed == {
        "children_by_label": {
            "A": [
                "Access to fresh produce",
                "Nutritional diversity",
            ],
            "H": [
                "Food budgeting practices",
                "Farmers' market presence",
                "Nutritional labeling accuracy",
            ],
        }
    }


def test_parse_generated_json_recovers_inline_children_mapping_with_mismatched_terminal_quote() -> (
    None
):
    parsed = hf_transformers._parse_generated_json(
        (
            "{'Availability of healthy foods' : ['Nutritional diversity', "
            "'Freshness of produce', 'Food accessibility\"]}"
        ),
        schema_name="children_by_label",
    )

    assert parsed == {
        "children_by_label": {
            "Availability of healthy foods": [
                "Nutritional diversity",
                "Freshness of produce",
                "Food accessibility",
            ]
        }
    }


def test_parse_generated_json_recovers_fenced_label_list_with_comments() -> None:
    response = (
        "```python\n"
        "[\n"
        "    'Urban sprawl and walkablility barriers',\n"
        "    'Nudging strategies in cafeteria/placemaking "
        "(e.g., salad bars over fritters in public spaces)',\n"
        '    \'"JUNK FOOD ZONES" (spatially concentrated fast-foodies '
        "vs supermarkets in low-SOC areas)',  # Geopolitical inequities\n"
        "    'Antibiotic overuse in livestock and gut-microbiome links "
        "to obesity/metabolic health',\n"
        "    'Climate-induced food price volatility "
        "(disproportionate impact on vulnerable eaters)'\n"
        "]\n"
        "```"
    )
    parsed = hf_transformers._parse_generated_json(response, schema_name="label_list")

    assert parsed == [
        "Urban sprawl and walkablility barriers",
        "Nudging strategies in cafeteria/placemaking "
        "(e.g., salad bars over fritters in public spaces)",
        '"JUNK FOOD ZONES" (spatially concentrated fast-foodies vs supermarkets in low-SOC areas)',
        "Antibiotic overuse in livestock and gut-microbiome links to obesity/metabolic health",
        "Climate-induced food price volatility (disproportionate impact on vulnerable eaters)",
    ]


def test_parse_generated_json_recovers_label_list_with_mismatched_internal_quote() -> None:
    parsed = hf_transformers._parse_generated_json(
        (
            "['Prevalence of green fields', 'Culture of eating', 'Consumers', "
            "'Public support for healthy products, 'Stress', 'Depression', "
            "'Marketing of unhealthy foods']"
        ),
        schema_name="label_list",
    )

    assert parsed == [
        "Prevalence of green fields",
        "Culture of eating",
        "Consumers",
        "Public support for healthy products",
        "Stress",
        "Depression",
        "Marketing of unhealthy foods",
    ]


def test_parse_generated_json_recovers_fenced_label_list_ignoring_trailing_note() -> None:
    response = (
        "```python\n"
        "[\n"
        "    'Government policies on urban greening',\n"
        "    'Biodiversification of recreational spaces',\n"
        "    'Influencer marketing impact on dietary choices'\n"
        "]```\n\n"
        "*Note: Adjusted to 3 due to constraints.*"
    )
    parsed = hf_transformers._parse_generated_json(response, schema_name="label_list")

    assert parsed == [
        "Government policies on urban greening",
        "Biodiversification of recreational spaces",
        "Influencer marketing impact on dietary choices",
    ]


def test_parse_generated_json_recovers_truncated_label_list_missing_closing_bracket() -> None:
    response = (
        "['Nutritive education',\n"
        "'Relationship between trails and obesity',\n"
        "'Social influence on food choices'"
    )

    parsed = hf_transformers._parse_generated_json(response, schema_name="label_list")

    assert parsed == [
        "Nutritive education",
        "Relationship between trails and obesity",
        "Social influence on food choices",
    ]


def test_parse_generated_json_normalizes_label_dict_with_packed_single_string() -> None:
    response = (
        "{\"labels\": [\"Nutritional education', 'Obesity prevention', "
        "'Food accessibility', 'Food policy', 'Health promotion\"]}"
    )

    parsed = hf_transformers._parse_generated_json(response, schema_name="label_list")

    assert parsed == [
        "Nutritional education",
        "Obesity prevention",
        "Food accessibility",
        "Food policy",
        "Health promotion",
    ]


def test_parse_generated_json_recovers_comma_separated_label_list_with_inconsistent_quotes() -> (
    None
):
    response = (
        "[\n"
        "  'Quality to hire',\n"
        "'Traillness impact on mental heath', \n"
        "'Relationship hildreness of food access', 'ealthy foods demand',"
        "Aesthetics in health"
    )

    parsed = hf_transformers._parse_generated_json(response, schema_name="label_list")

    assert parsed == [
        "Quality to hire",
        "Traillness impact on mental heath",
        "Relationship hildreness of food access",
        "ealthy foods demand",
        "Aesthetics in health",
    ]


def test_parse_generated_json_recovers_single_bare_label_list_item() -> None:
    response = "[\n  (Public support &amp; healthy eating],"

    parsed = hf_transformers._parse_generated_json(response, schema_name="label_list")

    assert parsed == ["(Public support &amp; healthy eating"]


def test_parse_generated_json_recovers_plain_scalar_label_for_label_list() -> None:
    parsed = hf_transformers._parse_generated_json(
        "Hypertension",
        schema_name="label_list",
    )

    assert parsed == ["Hypertension"]


def test_parse_generated_json_recovers_bare_comma_separated_label_list() -> None:
    parsed = hf_transformers._parse_generated_json(
        "Cholecystokinin, Metabolic syndrome",
        schema_name="label_list",
    )

    assert parsed == ["Cholecystokinin", "Metabolic syndrome"]


def test_parse_generated_json_recovers_commented_label_list_with_trailing_markdown_note() -> None:
    response = (
        "['Lifestyle interventions (diet/exercise programs)',\n"
        "'Hepatic steatosis (NAFL/NASH progression)', # From obesity/MetS links\n"
        "'Bone marrow adiposity and hematopoiesis dysfunction', # Emerging obesity-metabol* research\n"  # noqa: E501
        "'Circadian misalignment (shift work/sleep disorders)' + metabolic disruption links, # Short sleep/obesity axis expansion]\n"  # noqa: E501
        "**Note:** Prioritized nodes with:\n"
        "1) relevance.\n"
    )

    parsed = hf_transformers._parse_generated_json(response, schema_name="label_list")

    assert parsed == [
        "Lifestyle interventions (diet/exercise programs)",
        "Hepatic steatosis (NAFL/NASH progression)",
        "Bone marrow adiposity and hematopoiesis dysfunction",
        "Circadian misalignment (shift work/sleep disorders)",
    ]


def test_normalize_label_list_payload_accepts_mapping_payload() -> None:
    class LabelMapping(Mapping[str, object]):
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def __getitem__(self, key: str) -> object:
            return self._payload[key]

        def __iter__(self):
            return iter(self._payload)

        def __len__(self) -> int:
            return len(self._payload)

        def get(self, key: str, default: object | None = None) -> object | None:
            return self._payload.get(key, default)

    parsed = hf_transformers._normalize_label_list_payload(
        LabelMapping({"labels": "Nutritional education', 'Food accessibility"})
    )

    assert parsed == ["Nutritional education", "Food accessibility"]


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


def test_runtime_factory_uses_flash_attention_for_olmo_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_calls: list[dict[str, object]] = []

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(model: str, **kwargs):
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
    monkeypatch.setattr(hf_transformers, "_flash_attention_available", lambda: True)

    factory = HFTransformersRuntimeFactory(hf_token="hf-token")

    _ = factory.profile_for_chat_model("allenai/Olmo-3-7B-Instruct")

    assert model_calls == [
        {
            "model": "allenai/Olmo-3-7B-Instruct",
            "token": "hf-token",
            "dtype": "bfloat16",
            "attn_implementation": "flash_attention_2",
            "trust_remote_code": True,
        }
    ]


def test_runtime_generation_overrides_enable_low_memory_for_olmo_contrastive() -> None:
    overrides = hf_transformers.runtime_generation_overrides(
        model="allenai/Olmo-3-7B-Instruct",
        decoding_config=DecodingConfig(algorithm="contrastive", penalty_alpha=0.2, top_k=4),
    )

    assert overrides == {"low_memory": True}


def test_should_disable_stateful_guard_only_for_qwen_contrastive() -> None:
    assert hf_transformers.should_disable_stateful_guard(
        model="Qwen/Qwen3.5-9B",
        decoding_config=DecodingConfig(algorithm="contrastive", penalty_alpha=0.2, top_k=4),
    )
    assert not hf_transformers.should_disable_stateful_guard(
        model="allenai/Olmo-3-7B-Instruct",
        decoding_config=DecodingConfig(algorithm="contrastive", penalty_alpha=0.2, top_k=4),
    )
    assert not hf_transformers.should_disable_stateful_guard(
        model="Qwen/Qwen3.5-9B",
        decoding_config=DecodingConfig(algorithm="greedy"),
    )


def test_parse_generated_json_recovers_children_mapping_with_comment_suffix() -> None:
    parsed = _parse_generated_json(
        "{'Marketing of healthy foods' : ['Nutrition communication', "
        "'$$$ natural product branding', 'Wellness promotion strategies'], "
        "/* explanatory comment */}",
        schema_name="children_by_label",
    )

    assert parsed == {
        "children_by_label": {
            "Marketing of healthy foods": [
                "Nutrition communication",
                "$$$ natural product branding",
                "Wellness promotion strategies",
            ]
        }
    }


def test_parse_generated_json_recovers_empty_children_mapping_before_commentary() -> None:
    parsed = _parse_generated_json(
        "{} // No recommendations as there are no input concepts to relate",
        schema_name="children_by_label",
    )

    assert parsed == {"children_by_label": {}}


def test_parse_generated_json_recovers_label_list_from_prose_with_quoted_candidates() -> None:
    parsed = _parse_generated_json(
        "Understood. Return proposal. "
        "'Healthy eating patterns', 'Physical inactivity', 'Tiredness', "
        "'High cholesterol', 'Food environment', 'A', 'Excess'",
        schema_name="label_list",
    )

    assert parsed == [
        "Healthy eating patterns",
        "Physical inactivity",
        "Tiredness",
        "High cholesterol",
        "Food environment",
    ]


def test_parse_generated_json_recovers_empty_children_mapping_with_nested_braces_in_commentary() -> (  # noqa: E501
    None
):
    """Regression: SSH1 {} // commentary containing {'A': [...]} nested blocks."""
    text = (
        "{} // No recommendations. "
        "So for A (Appetite), produce five related concepts:\n\n"
        "{\n  A: ['Cravings', 'Nutrition', 'Hunger']\n}"
    )
    parsed = _parse_generated_json(text, schema_name="children_by_label")

    assert parsed == {"children_by_label": {}}


def test_parse_generated_json_recovers_children_mapping_with_missing_opening_quotes() -> None:
    """Regression: SSH2 children dict with missing opening quotes on some values."""
    text = (
        "{\n"
        "  'body dissatisfaction': [\n"
        "    'self-image concern',\n"
        "   ideal weight beliefs',\n"
        "    social comparison effects'\n"
        "  ]\n"
        "} \n\n"
        "Wait the user said for each of the names.\n\n"
        "{'body dissatisfaction' : ['ideal body image', 'physical scrutiny']}"
    )
    parsed = _parse_generated_json(text, schema_name="children_by_label")

    assert parsed == {
        "children_by_label": {
            "body dissatisfaction": [
                "self-image concern",
                "ideal weight beliefs",
                "social comparison effects",
            ]
        }
    }


def test_parse_generated_json_recovers_children_mapping_with_commentary_and_second_attempt() -> (
    None
):
    """Regression: SSH1 partial dict + parenthesized note + second dict attempt."""
    text = (
        "{\n"
        "  'Quality of equipment': ['Maintenance level', 'Competitiveness'],  \n"
        "  (Note: Only two were generated as an example.)\n"
        "  }  \n"
        "  {\n"
        "    'Safety standards': ['Injury prevention', 'Regulatory compliance', "
        "'Weather preparedness']\n"
        "  }"
    )
    parsed = _parse_generated_json(text, schema_name="children_by_label")

    assert isinstance(parsed, dict)
    assert "children_by_label" in parsed
    mapping = parsed["children_by_label"]
    assert isinstance(mapping, dict)
    assert len(mapping) >= 1


def test_parse_generated_json_skips_unparseable_tuples_in_edge_list() -> None:
    """Regression: SSH1 edge_list with single-part tuples like (as stress affects X)."""
    text = (
        "[(Stress, Mental well-being), "
        "(as stress affects mental_well_being), "
        "(Appetite, Eating habits)]"
    )
    parsed = _parse_generated_json(text, schema_name="edge_list")

    assert parsed == [
        ("Stress", "Mental well-being"),
        ("Appetite", "Eating habits"),
    ]


def test_parse_generated_json_returns_none_for_only_unparseable_tuples() -> None:
    """When all tuples are single-part, fall through to next recovery."""
    text = "(as stress affects mental_well_being)"

    with pytest.raises(ValueError, match="Model did not return valid structured output"):
        _parse_generated_json(text, schema_name="edge_list")


def test_parse_generated_json_recovers_label_list_from_unquoted_prose_candidates() -> None:
    """Regression: SSH3 label_list with unquoted candidates in prose."""
    text = (
        "Return only the  'Body image', 'Weight stigma', "
        "'Weight-based discrimination', 'Health impact of obesity', "
        "'Mental health', 'Weight-related quality of life'"
    )
    parsed = _parse_generated_json(text, schema_name="label_list")

    assert parsed == [
        "Body image",
        "Weight stigma",
        "Weight-based discrimination",
        "Health impact of obesity",
        "Mental health",
    ]


def test_parse_generated_json_recovers_children_mapping_with_paren_list_closer() -> None:
    """Regression: OLMo produces ) instead of ] as list closer."""
    text = (
        "{\n"
        '  "Stress": ["anxiety reduction", \n'
        '              "calmness", \n'
        '              "resilience"),\n'
        '  "Workload": ["intensity", "crisis"]\n'
        "}"
    )
    parsed = _parse_generated_json(text, schema_name="children_by_label")

    assert parsed == {
        "children_by_label": {
            "Stress": ["anxiety reduction", "calmness", "resilience"],
            "Workload": ["intensity", "crisis"],
        }
    }


def test_parse_generated_json_recovers_remote_unquoted_children_key() -> None:
    text = "{Weight bias: ['Body shyness', 'Muscle dysmorphia']"

    parsed = _parse_generated_json(text, schema_name="children_by_label")

    assert parsed == {
        "children_by_label": {
            "Weight bias": ["Body shyness", "Muscle dysmorphia"],
        }
    }


def test_parse_generated_json_recovers_remote_unquoted_children_key_with_nonstrings() -> None:
    text = "{nutritional awareness: ['dietary knowledge', [1, 2, 3], [4, 5, 6]]"

    parsed = _parse_generated_json(text, schema_name="children_by_label")

    assert parsed == {
        "children_by_label": {
            "nutritional awareness": ["dietary knowledge"],
        }
    }


def test_parse_generated_json_recovers_remote_malformed_terminal_key_quote() -> None:
    text = '{nutritional awareness": ["dietary knowledge", "nutri-literacymythbusting"'

    parsed = _parse_generated_json(text, schema_name="children_by_label")

    assert parsed == {
        "children_by_label": {
            "nutritional awareness": ["dietary knowledge", "nutri-literacymythbusting"],
        }
    }


def test_parse_generated_json_recovers_json_encoded_children_mapping_string() -> None:
    text = "\"{Weight bias: ['Body shyness', 'Muscle dysmorphia']\""

    parsed = _parse_generated_json(text, schema_name="children_by_label")

    assert parsed == {
        "children_by_label": {
            "Weight bias": ["Body shyness", "Muscle dysmorphia"],
        }
    }


def test_parse_generated_json_recovers_unicode_punctuation_children_mapping_string() -> None:
    text = '"{医院视\\uff1a [“muscular ideal”, 流\\u884c体采形”, 👤类型✍️,“hyperthrophy”]}\\'

    parsed = _parse_generated_json(text, schema_name="children_by_label")

    assert parsed == {
        "children_by_label": {
            "医院视": [
                "muscular ideal",
                "流行体采形",
                "👤类型✍️",
                "hyperthrophy",
            ],
        }
    }


def test_parse_generated_json_ignores_dangling_terminal_quote_in_children_list() -> None:
    text = "{Control: ['Autonomy', 'Complianc',']}"

    parsed = _parse_generated_json(text, schema_name="children_by_label")

    assert parsed == {
        "children_by_label": {
            "Control": ["Autonomy", "Complianc"],
        }
    }


def test_parse_generated_json_recovers_quoted_bracket_edge_pairs() -> None:
    """Regression: edge pairs using ['Source','Target'] with quotes."""
    text = (
        "Return proposed edges as: ['Weight bias','Socio-economic status'] "
        "not, do not add anything else"
    )
    parsed = _parse_generated_json(text, schema_name="edge_list")

    assert parsed == [("Weight bias", "Socio-economic status")]


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


def test_parse_generated_json_recovers_markdown_code_fence_only() -> None:
    """Regression: Qwen contrastive returns just ```json instead of JSON."""
    parsed = _parse_generated_json("```json", schema_name="children_by_label")
    assert parsed == {"children_by_label": {}}


def test_parse_generated_json_recovers_literal_error_text() -> None:
    """Regression: Qwen contrastive returns literal 'Error' as raw_text."""
    parsed = _parse_generated_json("Error", schema_name="children_by_label")
    assert parsed == {"children_by_label": {}}


def test_parse_generated_json_recovers_bare_word_in_brackets() -> None:
    """Regression: Qwen returns [diet] bare word in brackets with trailing garbage."""
    text = '{ "#AteWhatINeed messaging": ["nutri-messaging", "calorie-tracker alerts", [diet] reminders"]}'  # noqa: E501
    parsed = _parse_generated_json(text, schema_name="children_by_label")
    assert parsed == {
        "children_by_label": {
            "#AteWhatINeed messaging": ["nutri-messaging", "calorie-tracker alerts"]
        }
    }


def test_parse_generated_json_recovers_nested_brackets_with_mismatched_quote() -> None:
    """Regression: Qwen returns nested brackets with mismatched quotes."""
    text = "{lack of energy: ['fatigue', [exhaustion', low vitality']}"
    parsed = _parse_generated_json(text, schema_name="children_by_label")
    # Sanitization removes malformed nested content, leaving just the valid first item
    assert parsed == {"children_by_label": {"lack of energy": ["fatigue"]}}


def test_parse_generated_json_recovers_unquoted_key_comma_separated_children() -> None:
    """Regression: Qwen returns unquoted key with comma-separated values (no brackets)."""
    text = "{Adaptability: Resilience, Flexibilite, Versatilit\u00e9agility}"
    parsed = _parse_generated_json(text, schema_name="children_by_label")
    # Parser recovers by treating the comma-separated part as values for the key
    assert parsed == {
        "children_by_label": {
            "Adaptability": ["Resilience", "Flexibilite", "Versatilit\u00e9agility"]
        }
    }


def test_parse_generated_json_recovers_json_pseudo_fence() -> None:
    """Regression: Qwen returns truncated ```json-pseudo marker instead of JSON."""
    text = "```json-pseudo (Note: JSON does n"
    parsed = _parse_generated_json(text, schema_name="children_by_label")
    assert parsed == {"children_by_label": {}}


def test_parse_generated_json_recovers_truncated_children_mapping_missing_close() -> None:
    """Regression: Qwen returns children mapping missing the closing ]}."""
    text = "{Anxiolytic: ['Sedative-hypnotic', 'Anticonvulsant'}"
    parsed = _parse_generated_json(text, schema_name="children_by_label")
    assert parsed == {"children_by_label": {"Anxiolytic": ["Sedative-hypnotic", "Anticonvulsant"]}}


def test_parse_generated_json_recovers_trailing_comma_before_brace() -> None:
    """Regression: Qwen returns children mapping with trailing comma before }."""
    text = "{Discipline: ['Patience', 'Consistenc', }]"
    parsed = _parse_generated_json(text, schema_name="children_by_label")
    assert parsed == {"children_by_label": {"Discipline": ["Patience", "Consistenc"]}}


def test_parse_generated_json_recovers_fenced_python_children_without_colons() -> None:
    """Regression: Mistral returns fenced python with children dict missing colons."""
    text = '```python\n{\n    "anorexia nervosa" ["starvation behavior", "extreme caloriavidence",\n                       "sevelfeedingrestrictiveness"]\n```'  # noqa: E501
    parsed = _parse_generated_json(text, schema_name="children_by_label")
    # Parser recovers by extracting quoted key-value pairs without requiring colons
    assert parsed == {
        "children_by_label": {
            "anorexia nervosa": [
                "starvation behavior",
                "extreme caloriavidence",
                "sevelfeedingrestrictiveness",
            ]
        }
    }


def test_parse_generated_json_recovers_embedded_fence_in_children_key() -> None:
    """Regression: Mistral contrastive embeds ```ng inside a JSON string key.

    The model generated: "Emotional eati\\n```ng" as a dict key, where \\n```ng
    looks like a code fence language specifier but is actually part of the key text.
    The key is recovered as "Emotional eati" after stripping the embedded fence.
    """
    text = '```python\n{\n    "Emotional eati\n```ng" : [\n    "mindful eating habits",\n\t"Arousal-triggered snack cravings",\n\t"Heuristic food choices under stress (HFUS) behavior",\n     "Strategic comfort-avoidance behaviors (SCAB)",\n     "Regulatory focus imbalance (RFI)"\n    ]\n}\n```'  # noqa: E501
    parsed = _parse_generated_json(text, schema_name="children_by_label")
    assert parsed == {
        "children_by_label": {
            "Emotional eati": [
                "mindful eating habits",
                "Arousal-triggered snack cravings",
                "Heuristic food choices under stress (HFUS) behavior",
                "Strategic comfort-avoidance behaviors (SCAB)",
                "Regulatory focus imbalance (RFI)",
            ]
        }
    }


def test_parse_generated_json_recovers_children_with_key_comments() -> None:
    """Regression: Mistral returns children with parenthetical comments in keys."""
    text = '```python\n{\n    "Supportive食品环境" (Chinese for clarity):\n    ["nutrient-abundant diet culture", "convenient health food access"]\n}\n```'  # noqa: E501
    parsed = _parse_generated_json(text, schema_name="children_by_label")
    assert parsed == {
        "children_by_label": {
            "Supportive食品环境": [
                "nutrient-abundant diet culture",
                "convenient health food access",
            ]
        }
    }


def test_parse_generated_json_recovers_children_with_single_quoted_key_and_comment() -> None:
    """Regression: Mistral single-quoted key with parenthetical comment."""
    text = "```python\n{\n    'Embodied joy' (opposite: anheonia):\n    ['Vibrational euphoria', 'Emotional resonance']\n}\n```"  # noqa: E501
    parsed = _parse_generated_json(text, schema_name="children_by_label")
    assert parsed == {
        "children_by_label": {"Embodied joy": ["Vibrational euphoria", "Emotional resonance"]}
    }


def test_parse_generated_json_rejects_truncated_edge_list() -> None:
    """Regression: Qwen returns very short truncated edge_list - too short to recover."""
    text = '[\n  ("Prevalene'
    with pytest.raises(ValueError, match="Model did not return valid structured output"):
        _parse_generated_json(text, schema_name="edge_list")


def test_parse_generated_json_recovers_embedded_fence_in_children_key_last_fence_logic() -> None:
    """Regression: Mistral contrastive embeds ``` inside a JSON string key.

    The model generated: "Emotional eati\\n```ng" as a dict key, where \\n```ng
    looks like a code fence language specifier but is actually part of the key text.
    The closing fence should be identified as the LAST ``` in the text, not the
    embedded one, so the key is recovered as "Emotional eati" (ng lost to embedded fence).
    """
    text = '```python\n{\n    "Emotional eati\n```ng" : [\n    "mindful eating habits",\n\t"Arousal-triggered snack cravings",\n\t"Heuristic food choices under stress (HFUS) behavior",\n     "Strategic comfort-avoidance behaviors (SCAB)",\n     "Regulatory focus imbalance (RFI)"\n    ]\n}\n```'  # noqa: E501
    parsed = _parse_generated_json(text, schema_name="children_by_label")
    assert parsed == {
        "children_by_label": {
            "Emotional eati": [
                "mindful eating habits",
                "Arousal-triggered snack cravings",
                "Heuristic food choices under stress (HFUS) behavior",
                "Strategic comfort-avoidance behaviors (SCAB)",
                "Regulatory focus imbalance (RFI)",
            ]
        }
    }


def test_parse_generated_json_recovers_thinking_block_inside_code_fence() -> None:
    """Regression: Qwen contrastive returns <think> thinking block inside code fence.

    The output starts with fenced Python but the model emits <think>...</think>
    thinking tags inside the block before the actual JSON content.
    These thinking tags must be stripped before JSON parsing.
    """
    text = (
        "```python\n"
        "{ <think>\n"
        "\"Okay, let's tackle this problem. The user wants me to come up with 3 related concept names for 'Preva...\n"  # noqa: E501
        "</think>\n"
        '    "Prevalence of walking trails": ["Trail density", "Path length per capita", "Trail accessibility"],\n'  # noqa: E501
        '    "Quality of sport infrastructure": ["Facility maintenance", "Equipment quality", "Safety standards"]\n'  # noqa: E501
        "}\n```"
    )
    parsed = _parse_generated_json(text, schema_name="children_by_label")
    assert parsed == {
        "children_by_label": {
            "Prevalence of walking trails": [
                "Trail density",
                "Path length per capita",
                "Trail accessibility",
            ],
            "Quality of sport infrastructure": [
                "Facility maintenance",
                "Equipment quality",
                "Safety standards",
            ],
        }
    }


def test_parse_generated_json_recovers_markdown_bold_in_children_values() -> None:
    """Regression: Mistral contrastive emits **bold** markdown inside fenced Python values.

    The model uses **text** bold markers inside list values, which are not valid
    JSON. These bold markers must be stripped from fenced content before parsing.
    """
    text = (
        "```python\n"
        "{\n"
        "    'Macronutrient balance (carbs : fats : proteïn)' : [\n"
        "        'Proportion of plant-sourced foods',\n"
        "        'Glycemic load per calorie',\n"
        "        '**Original concept: Nutrient density in diets (per calorie)** :\n"
        "        [   '**1. Protein efficiency ratio**',\n"
        "            '**2. Antioxidant-capacity-to-inflammasome-priming**' ]\n"
        "}\n```"
    )
    parsed = _parse_generated_json(text, schema_name="children_by_label")
    assert parsed == {
        "children_by_label": {
            "Macronutrient balance (carbs : fats : proteïn)": [
                "Proportion of plant-sourced foods",
                "Glycemic load per calorie",
                "Original concept: Nutrient density in diets (per calorie)",
                "1. Protein efficiency ratio",
                "2. Antioxidant-capacity-to-inflammasome-priming",
            ]
        }
    }


def test_parse_generated_json_recovers_fenced_python_with_thinking_block() -> None:
    """Regression: Qwen algo3 contrastive has <think> block at start of fenced output."""
    text = '```python\n{ <think>\n"Okay, let\'s tackle this problem..."\n</think>\n"RelatedConceptA": ["Sub1", "Sub2"],\n"RelatedConceptB": ["Sub3"]\n}\n```'  # noqa: E501
    parsed = _parse_generated_json(text, schema_name="children_by_label")
    assert parsed == {
        "children_by_label": {
            "RelatedConceptA": ["Sub1", "Sub2"],
            "RelatedConceptB": ["Sub3"],
        }
    }


def test_parse_generated_json_rejects_truncated_edge_list_with_partial_second_element() -> None:
    """Regression: Qwen returns edge_list with first element complete but second truncated."""
    text = "[('Marketing of healthy foods', 'Cul"
    with pytest.raises(ValueError, match="Model did not return valid structured output"):
        _parse_generated_json(text, schema_name="edge_list")


def test_parse_generated_json_rejects_thinking_text_output() -> None:
    """Regression: Model returns thinking/reasoning text instead of structured output."""
    text = "{ <think>\n\"Okay, let's tackle this problem. The user wants me to come up with 3 related concept names for 'Preva...\n</think>\n"  # noqa: E501
    with pytest.raises(ValueError, match="Model did not return valid structured output"):
        _parse_generated_json(text, schema_name="edge_list")


def test_parse_generated_json_recovers_children_double_quoted_single_quoted_key_with_comment() -> (
    None
):
    """Regression: Mistral returns "'key' (comment): [values]" pattern."""
    text = """```python
{
    "'jun food dominance index' (JFD*): [
        "processed food saturation index (PFSI)",
        "ultra-refined calorie ratio (URC*R)"
    ]
}
```"""
    parsed = _parse_generated_json(text, schema_name="children_by_label")
    assert parsed == {
        "children_by_label": {
            "'jun food dominance index'": [
                "processed food saturation index (PFSI)",
                "ultra-refined calorie ratio (URC*R)",
            ]
        }
    }


def test_parse_generated_json_recovers_first_valid_entry_from_markdown_heavy_fenced_children_mapping() -> (  # noqa: E501
    None
):
    text = """```python {
    'Macronutrient balance (carbs : fats : proteïn)' : ['Proportion of plant-sourced foods',
                                                     'Glycemic load per calorie',
                                                     '"Empty-calorie" intake (e.g., sugary drinks/snacks vs nutrient-packed)',
                                                     '% calories sourced post-digestive satiety (fructooligosaccharides, resistant starçh) vs pre-digéstive (glucose/fructoseloaded)',
                                                     'Micelle-worthy fat-to-solubility ratio (phytosterols, lecithin vs trans/saturated)' ],

    **Original concept: Nutrient density in dïets (per c̄alore)** :
    [   **1. Protein efficiency ratio (P:kcal, g/1e3kc) vs metabolic flexibility (insulin index + amino-acid scoring)'**
        **2. Antioxidant-capacity-to-inflammasome-priming (ORAC vs NF-κB/CRP ratios, mg/TRIM30α-equivalenţ)**,
        **2.b (split if too dense) "Polyphenol:proanthocyanidin (flavan- vs stilbenoid)-dominance" +** 'Nitric oxide bioavailabiĺity (L-arginine-to-citrulline + eN0S/ADMAS) per 100g' **]**
        **— (Note: Above 2/2.5 are sub-concepts of nutrient "efficiency" not just density.)** —

        **Alternative 3-5 (broader, non-mutually-excluṣive):**
        **3. FODMAP:SCFAligand ratio (fermentable vs butyrate-progenitor prebiotics, g/L short-chain-fatty-acidexcrȩtioŋ)**,
        **4. Trace-mineral-biodisponibility (Zn/Cupremic vs phytate chelatation, μmoI/L urinary excretiōn post-chalconefortiﬁcātiōn)**,
        **5. Lipid-partition-coefᶠcient (DHA/EpDHA in n-3 PUFA vs arachidonic/γ-linolenic, % bound to chylomicrons/HDL)**
}
```"""  # noqa: E501

    parsed = _parse_generated_json(text, schema_name="children_by_label")

    assert parsed == {
        "children_by_label": {
            "Macronutrient balance (carbs : fats : proteïn)": [
                "Proportion of plant-sourced foods",
                "Glycemic load per calorie",
                '"Empty-calorie" intake (e.g., sugary drinks/snacks vs nutrient-packed)',
                "% calories sourced post-digestive satiety (fructooligosaccharides, resistant starçh) vs pre-digéstive (glucose/fructoseloaded)",  # noqa: E501
                "Micelle-worthy fat-to-solubility ratio (phytosterols, lecithin vs trans/saturated)",  # noqa: E501
            ]
        }
    }


def test_recover_fenced_python_children_mapping_mistral_quote_artifact():
    payload = r"""```python{
    'Pressure_to_be_thin': [
        'Body_dissatisfaction_with_standarized_beautystandards', # Lower if satisfaction increases, higher if it diminishetheself-worth tied to weight/shapelooks
        'Access_to_nutritious_meals_affordabiltiy_gapp', # Inversely related—higher access reduces pressure to restrict; gaps worsenextremestratagies (e.g.fastingskippingmealsafeffectively)
        'Bodyshaming_exposurereinforcemechanisms_on_sociomedia_platforms', #'More' = amplified exposure (algorithmic curation,tiktokchallengescollabwitheditors); 'lessthroughregulatiocontentpolicymoderation
            'Perfecitonism_towardexternal_aestheticheirarcan_judgmenr', # Quantifiable via scales like "Ifeelobligatedtopreftheidealofthemostfavoredbodies"scored0(none)to1(compulsiveadherence)
            '"Thinspiration"consumption_influencemagnitudoftrendinghashtagex: "#SkinnyBeforeAfter"or"IWoreASmallerClothesizeThisMorning'",
            'Lack_offinancial_stabillity_linktodietrestricting_behaviors' # Correlate: "DoIavoidbuyinggroceriesbecausemoneyisshort?Yes(5)No(1).Higherfinancialstress→morereliedoncheapoflimitedcaloriestoeat"
    ]
}
```"""  # noqa: E501
    from llm_conceptual_modeling.common.hf_transformers import _parse_generated_json

    result = _parse_generated_json(payload, schema_name="children_by_label")
    assert "Pressure_to_be_thin" in result["children_by_label"]
    assert len(result["children_by_label"]["Pressure_to_be_thin"]) == 6
