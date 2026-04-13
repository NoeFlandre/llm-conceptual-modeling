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
