from collections.abc import Mapping

from llm_conceptual_modeling.common.types import Edge

Thesaurus = Mapping[str, Mapping[str, list[str]]]


def build_term_normalizer(thesaurus: Thesaurus) -> dict[str, str]:
    normalizer: dict[str, str] = {}
    _register_term_family(normalizer, thesaurus.get("synonyms", {}))
    _register_term_family(normalizer, thesaurus.get("antonyms", {}))

    return normalizer


def normalize_edge_terms(edges: list[Edge], thesaurus: Thesaurus) -> list[Edge]:
    normalizer = build_term_normalizer(thesaurus)
    normalized_edges: list[Edge] = []

    for source_term, target_term in edges:
        normalized_source = normalizer.get(source_term, source_term)
        normalized_target = normalizer.get(target_term, target_term)
        normalized_edge = (normalized_source, normalized_target)
        normalized_edges.append(normalized_edge)

    return normalized_edges


def _register_base_term(normalizer: dict[str, str], base_term: str) -> None:
    normalizer[base_term] = base_term


def _register_variants(normalizer: dict[str, str], base_term: str, variants: list[str]) -> None:
    for variant in variants:
        normalizer[variant] = base_term


def _register_term_family(
    normalizer: dict[str, str],
    term_family: Mapping[str, list[str]],
) -> None:
    for base_term, variants in term_family.items():
        _register_base_term(normalizer, base_term)
        _register_variants(normalizer, base_term, variants)
