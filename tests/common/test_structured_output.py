import pytest

import llm_conceptual_modeling.common.structured_output as structured_output
from llm_conceptual_modeling.common.structured_output import normalize_structured_response


def test_normalize_edge_list_drops_short_odd_flat_label_list() -> None:
    normalized = normalize_structured_response(
        [
            "Prevalence of green fields",
            "Culture of eating",
            "Consumers",
            "Public support for healthy products",
            "Stress",
            "Depression",
            "Marketing of unhealthy foods",
        ],
        schema_name="edge_list",
    )

    assert normalized == {"edges": []}


def test_normalize_edge_list_pairs_flat_scalar_endpoints() -> None:
    normalized = normalize_structured_response(
        ["source", "target", "another source", "another target"],
        schema_name="edge_list",
    )

    assert normalized == {
        "edges": [
            {"source": "source", "target": "target"},
            {"source": "another source", "target": "another target"},
        ]
    }


def test_normalize_edge_list_drops_noisy_dangling_endpoint() -> None:
    normalized = normalize_structured_response(
        ["source", "target", "another source", "another target", "[metadata, note]"],
        schema_name="edge_list",
    )

    assert normalized == {
        "edges": [
            {"source": "source", "target": "target"},
            {"source": "another source", "target": "another target"},
        ]
    }


def test_normalize_edge_list_rejects_unrecoverable_odd_endpoint_list() -> None:
    with pytest.raises(ValueError, match="even number of items"):
        normalize_structured_response(
            ["source", "target", "dangling"],
            schema_name="edge_list",
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"source": "source", "target": "target"},
        ("source", "target"),
    ],
)
def test_normalize_edge_list_accepts_mapping_and_tuple_items(payload: object) -> None:
    assert normalize_structured_response([payload], schema_name="edge_list") == {
        "edges": [{"source": "source", "target": "target"}]
    }


@pytest.mark.parametrize(
    "payload",
    [["source"], "not an edge", {"source": "source"}],
)
def test_normalize_edge_list_rejects_invalid_items(payload: object) -> None:
    with pytest.raises(
        ValueError,
        match="Invalid edge item shape|even number of items|null edge target",
    ):
        normalize_structured_response([payload], schema_name="edge_list")


def test_normalize_children_by_label_mapping_payload() -> None:
    normalized = normalize_structured_response(
        {
            "children_by_label": {
                "Valid parent": ["Valid child"],
            }
        },
        schema_name="children_by_label",
    )

    assert normalized == {
        "children_by_label": {
            "Valid parent": ["Valid child"],
        }
    }


def test_normalize_children_by_label_single_tuple_payload() -> None:
    normalized = normalize_structured_response(
        ("Valid parent", ["Valid child"]),
        schema_name="children_by_label",
    )

    assert normalized == {
        "children_by_label": {
            "Valid parent": ["Valid child"],
        }
    }


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            [["Parent", ["Child", None, "none", "  ", 7]]],
            {"Parent": ["Child", "7"]},
        ),
        ([['Parent', None]], {"Parent": []}),
        ([['Parent', "Child"]], {"Parent": ["Child"]}),
        (["Parent", None], {"Parent": []}),
        (["Parent", "Child"], {"Parent": ["Child"]}),
        ([], {}),
    ],
)
def test_normalize_children_by_label_sequence_payload_variants(
    payload: list[object],
    expected: dict[str, list[str]],
) -> None:
    normalized = normalize_structured_response(
        payload,
        schema_name="children_by_label",
    )

    assert normalized == {"children_by_label": expected}


@pytest.mark.parametrize(
    "payload",
    [
        [["Parent"]],
        ["Parent", "Child", "Unexpected extra item"],
        [{"Parent": "Child"}, "Child"],
    ],
)
def test_normalize_children_by_label_rejects_invalid_sequence_payload(
    payload: list[object],
) -> None:
    with pytest.raises(ValueError, match="Unsupported structured response shape"):
        normalize_structured_response(
            payload,
            schema_name="children_by_label",
        )


def test_normalize_children_by_label_rejects_null_parent_in_sequence_payload() -> None:
    with pytest.raises(ValueError, match="null parent label"):
        normalize_structured_response(
            [None, "Child"],
            schema_name="children_by_label",
        )


@pytest.mark.parametrize(
    ("schema_name", "payload", "expected"),
    [
        (
            "edge_list",
            {"edges": [["source", "target"]]},
            {"edges": [{"source": "source", "target": "target"}]},
        ),
        ("vote_list", {"votes": ["Y", 0]}, {"votes": ["Y", "0"]}),
        ("label_list", {"labels": ["A", "B"]}, {"labels": ["A", "B"]}),
        ("vote_list", ["Y", "N"], {"votes": ["Y", "N"]}),
        ("label_list", ("A", "B"), {"labels": ["A", "B"]}),
        ("other", {"value": 1}, {"value": 1}),
    ],
)
def test_normalize_structured_response_schema_variants(
    schema_name: str,
    payload: object,
    expected: dict[str, object],
) -> None:
    assert normalize_structured_response(payload, schema_name=schema_name) == expected


def test_normalize_structured_response_preserves_mapping_without_expected_field() -> None:
    assert normalize_structured_response(
        {"other": 1},
        schema_name="edge_list",
    ) == {"other": 1}


@pytest.mark.parametrize(
    ("schema_name", "payload", "message"),
    [
        ("edge_list", {"edges": "not a list"}, "list of edges"),
        ("vote_list", {"votes": "not a list"}, "list of votes"),
        ("label_list", {"labels": "not a list"}, "list of labels"),
    ],
)
def test_normalize_structured_response_rejects_non_list_fields(
    schema_name: str,
    payload: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_structured_response(payload, schema_name=schema_name)


def test_normalize_structured_response_rejects_unsupported_scalar_payload() -> None:
    with pytest.raises(ValueError, match="Unsupported structured response shape"):
        normalize_structured_response(42, schema_name="edge_list")


def test_normalize_structured_response_reports_exact_unsupported_shape() -> None:
    with pytest.raises(
        ValueError,
        match="^Unsupported structured response shape for schema edge_list: int$",
    ):
        normalize_structured_response(42, schema_name="edge_list")


@pytest.mark.parametrize(
    ("schema_name", "field_name"),
    [("edge_list", "edges"), ("vote_list", "votes"), ("label_list", "labels")],
)
def test_normalize_structured_response_reports_exact_mapping_field_errors(
    schema_name: str,
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            f"^Structured {schema_name} response must contain a list of "
            f"{field_name}$"
        ),
    ):
        normalize_structured_response(
            {field_name: "not a list"},
            schema_name=schema_name,
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([{"source": None, "target": "target"}], "null edge source"),
        ([{"source": "source", "target": None}], "null edge target"),
        ([None], "null vote"),
    ],
)
def test_normalize_structured_response_reports_exact_null_item_errors(
    payload: list[object],
    message: str,
) -> None:
    schema_name = "edge_list" if "edge" in message else "vote_list"
    with pytest.raises(
        ValueError,
        match=f"^Structured response returned a {message}$",
    ):
        normalize_structured_response(payload, schema_name=schema_name)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([[None, "target"]], "null edge source"),
        ([['source', None]], "null edge target"),
    ],
)
def test_normalize_tuple_edge_items_report_exact_null_endpoint_errors(
    payload: list[object],
    message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"^Structured response returned a {message}$",
    ):
        normalize_structured_response(payload, schema_name="edge_list")


def test_normalize_structured_response_reports_label_and_parent_errors() -> None:
    with pytest.raises(
        ValueError,
        match="^Structured response returned a null parent label$",
    ):
        normalize_structured_response(
            {"children_by_label": {None: ["child"]}},
            schema_name="children_by_label",
        )

    with pytest.raises(
        ValueError,
        match="^Structured response returned a null label$",
    ):
        normalize_structured_response({"labels": [None]}, schema_name="label_list")

    with pytest.raises(
        ValueError,
        match="^Structured response returned an empty child label$",
    ):
        structured_output._normalize_children_value("")

    with pytest.raises(
        ValueError,
        match="^Structured response returned a null parent label$",
    ):
        normalize_structured_response(
            [[None, ["child"]]],
            schema_name="children_by_label",
        )


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        (None, False),
        (True, False),
        ("source", True),
        (7, True),
        ([], False),
        ({}, False),
    ],
)
def test_is_scalar_edge_endpoint_rejects_structured_values(
    item: object,
    expected: bool,
) -> None:
    assert structured_output._is_scalar_edge_endpoint(item) is expected


def test_noisy_dangling_edge_endpoint_thresholds_are_explicit() -> None:
    assert not structured_output._looks_like_noisy_dangling_edge_endpoint("")
    assert not structured_output._looks_like_noisy_dangling_edge_endpoint("plain")
    assert structured_output._looks_like_noisy_dangling_edge_endpoint("a,b,c")
    assert structured_output._looks_like_noisy_dangling_edge_endpoint("[metadata]")
    assert structured_output._looks_like_noisy_dangling_edge_endpoint("[metadata")
    assert structured_output._looks_like_noisy_dangling_edge_endpoint("metadata]")
    assert structured_output._looks_like_noisy_dangling_edge_endpoint("{metadata")
    assert structured_output._looks_like_noisy_dangling_edge_endpoint("metadata}")

    short_noisy = ["a", "b", "c", "[metadata]"]
    assert structured_output._drop_dangling_noisy_edge_endpoint(short_noisy) is None

    even_noisy = ["a", "b", "c", "d", "e", "[metadata]"]
    assert structured_output._drop_dangling_noisy_edge_endpoint(even_noisy) is None

    nine_items = ["a", "b", "c", "d", "e", "f", "g", "h", "[metadata]"]
    assert structured_output._drop_dangling_noisy_edge_endpoint(nine_items) == nine_items[:-1]


@pytest.mark.parametrize(
    ("item", "expected"),
    [("text", False), (b"text", False), (bytearray(b"text"), False), ([], True)],
)
def test_is_sequence_payload_excludes_textual_scalars(
    item: object,
    expected: bool,
) -> None:
    assert structured_output._is_sequence_payload(item) is expected


@pytest.mark.parametrize("item", ["", "none", "NONE"])
def test_normalize_string_item_rejects_empty_and_none_values(item: str) -> None:
    with pytest.raises(
        ValueError,
        match="^Structured response returned an empty label$",
    ):
        structured_output._normalize_string_item(item, "label")


def test_normalize_flat_edge_endpoints_reports_exact_errors() -> None:
    with pytest.raises(
        ValueError,
        match="^Structured response returned an empty edge endpoint$",
    ):
        structured_output._normalize_flat_edge_endpoints(["source", "target", ""])

    with pytest.raises(
        ValueError,
        match=(
            "^Structured edge_list flat string response must contain an even number "
            "of items$"
        ),
    ):
        structured_output._normalize_flat_edge_endpoints(["source", "target", "dangling"])
