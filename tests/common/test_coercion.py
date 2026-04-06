from llm_conceptual_modeling.common.io import coerce_int


def test_coerce_int_returns_default_for_malformed_values() -> None:
    assert coerce_int("false") == 0
    assert coerce_int(None) == 0
    assert coerce_int(True) == 0
    assert coerce_int("false", default=7) == 7
    assert coerce_int(False, default=7) == 7


def test_coerce_int_preserves_valid_integer_inputs() -> None:
    assert coerce_int(3) == 3
    assert coerce_int("11") == 11
