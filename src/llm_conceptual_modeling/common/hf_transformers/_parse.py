from __future__ import annotations

import ast
import json
import re

from llm_conceptual_modeling.common.hf_transformers._children_mapping import (
    _looks_like_children_mapping,
    _recover_children_mapping_from_lines,
    _recover_children_mapping_from_outer_block,
    _recover_double_quoted_children_values,
    _recover_fenced_python_children_mapping,
    _recover_inline_children_mapping,
    _recover_malformed_children_mapping,
    _recover_truncated_children_mapping_blocks,
    _recover_unquoted_key_comma_separated,
    _remove_nonstring_bracket_patterns,
    _sanitize_children_mapping_text_for_recovery,
    _strip_fenced_content_artifacts,
)
from llm_conceptual_modeling.common.hf_transformers._edge_list import (
    _extract_recoverable_edge_endpoints,
    _looks_like_truncated_single_edge_endpoint,
    _recover_bare_comma_separated_edge_pair,
    _recover_bracketed_edge_pairs,
)
from llm_conceptual_modeling.common.hf_transformers._label_list import (
    _normalize_label_list_payload,
    _recover_bare_comma_separated_label_list,
    _recover_label_list_from_lines,
    _recover_quoted_label_list_with_comments,
    _recover_single_bare_label,
)
from llm_conceptual_modeling.common.hf_transformers._policy import DecodingConfig


def _parse_generated_json(text: str, *, schema_name: str) -> object:
    stripped = _strip_code_fence(_strip_assistant_prefix(text.strip()))
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        recovered = _recover_non_json_response(text=stripped, schema_name=schema_name)
        if recovered is not None:
            return recovered
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError) as error:
            raise ValueError(f"Model did not return valid structured output: {text}") from error
    return _normalize_schema_response(parsed, schema_name=schema_name)


def _strip_assistant_prefix(text: str) -> str:
    lowered = text.lower()
    for prefix in ("assistant\n", "assistant:\n", "assistant: ", "assistant "):
        if lowered.startswith(prefix):
            return text[len(prefix) :].strip()
    return _strip_assistant_noise(text, lowered)


def _strip_assistant_noise(text: str, lowered: str) -> str:
    if lowered.startswith("assistant"):
        suffix = text[len("assistant") :].strip()
        if suffix and all(not character.isalnum() for character in suffix):
            return ""
    return text


def _strip_code_fence(text: str) -> str:
    """Strip markdown code fence markers from text.

    Uses non-greedy .*? to find the first ``` closing fence. If the
    extracted body has an unclosed JSON string (odd quote count), the
    closing fence was embedded inside a string key/value. In that case,
    re-extracts using the LAST ``` as the true closing fence.
    """
    fenced = re.search(r"```(?P<lang>[A-Za-z0-9_-]*)\s*(?P<body>.*?)```", text, flags=re.DOTALL)
    if fenced is None:
        return text
    body = fenced.group("body").strip()
    # If body has odd quote count, the closing fence was embedded inside a JSON
    # string. Re-extract using the LAST ``` as the true closing fence.
    if body.count('"') % 2 == 1:
        last_fence_pos = text.rfind("```")
        body_start = fenced.start("body")
        body = text[body_start:last_fence_pos].strip()
    return body


def _normalize_schema_response(parsed: object, *, schema_name: str) -> object:
    if isinstance(parsed, str):
        recovered = _recover_non_json_response(text=parsed, schema_name=schema_name)
        if recovered is not None:
            return recovered
    return _normalize_non_string_schema_response(parsed, schema_name=schema_name)


def _normalize_non_string_schema_response(parsed: object, *, schema_name: str) -> object:
    if schema_name == "label_list":
        recovered_labels = _normalize_label_list_payload(parsed)
        if recovered_labels is not None:
            return recovered_labels
    if schema_name == "children_by_label" and _looks_like_children_mapping(parsed):
        return {"children_by_label": parsed}
    return parsed


def _looks_retryable_malformed_output(*, text: str, schema_name: str) -> bool:
    stripped = _strip_code_fence(_strip_assistant_prefix(text.strip()))
    if not stripped:
        return True
    if "<think>" in stripped.lower():
        return True
    if schema_name == "edge_list":
        return _looks_retryable_edge_list_output(stripped)
    if schema_name == "children_by_label":
        return _looks_retryable_children_mapping_output(stripped)
    return False


def _looks_retryable_edge_list_output(text: str) -> bool:
    if text.startswith("[") and not text.rstrip().endswith("]"):
        return True
    quoted_items = re.findall(r"""['"]([^'"]+)['"]""", text)
    return bool(quoted_items) and len(quoted_items) % 2 == 1


def _looks_retryable_children_mapping_output(text: str) -> bool:
    return text.startswith("{") and not text.rstrip().endswith("}")


def _looks_retryable_normalization_failure(
    *,
    parsed_content: object,
    schema_name: str,
    error: ValueError,
) -> bool:
    if schema_name != "edge_list":
        return False
    if "even number of items" not in str(error):
        return False
    return isinstance(parsed_content, list) and all(
        isinstance(item, str) for item in parsed_content
    )


def _resolve_malformed_output_retry_limit(
    *,
    model: str,
    decoding_config: DecodingConfig,
    schema_name: str,
) -> int:
    if (
        model == _QWEN_CHAT_MODEL
        and decoding_config.algorithm == "contrastive"
        and schema_name in {"edge_list", "children_by_label"}
    ):
        return 3
    return 1


def _should_normalize_exhausted_malformed_edge_list_to_empty(
    *,
    model: str,
    decoding_config: DecodingConfig,
    schema_name: str,
    malformed_output_retries: int,
    malformed_output_retry_limit: int,
    text: str,
) -> bool:
    if model != _QWEN_CHAT_MODEL:
        return False
    if decoding_config.algorithm != "contrastive":
        return False
    if schema_name != "edge_list":
        return False
    if malformed_output_retries < malformed_output_retry_limit:
        return False
    return _looks_like_truncated_single_edge_endpoint(text)


def _recover_non_json_response(*, text: str, schema_name: str) -> object | None:
    stripped = text.strip().lower()
    if schema_name == "children_by_label":
        recovered_children = _recover_children_non_json_response(text, stripped)
        if recovered_children is not None:
            return {"children_by_label": recovered_children}
        return None
    return _recover_non_children_response(text, schema_name)


def _recover_non_children_response(text: str, schema_name: str) -> object | None:
    if schema_name == "label_list":
        return _recover_label_list_response(text)
    if schema_name == "edge_list":
        return _recover_edge_list_response(text)
    if schema_name == "vote_list":
        return _recover_vote_list_response(text)
    return None


def _recover_children_non_json_response(
    text: str,
    stripped: str,
) -> dict[str, list[str]] | None:
    if _is_empty_children_response(stripped):
        return {}
    recovered = _recover_first_children_candidate(text)
    if recovered is not None:
        return recovered
    return _recover_truncated_children_candidates(text)


def _recover_first_children_candidate(text: str) -> dict[str, list[str]] | None:
    for candidate_text in _children_recovery_candidates(text):
        recovered = _recover_children_candidate(candidate_text)
        if recovered is not None:
            return recovered
    return None


def _recover_truncated_children_candidates(text: str) -> dict[str, list[str]] | None:
    for candidate_text in _recover_truncated_children_mapping_blocks(text):
        if candidate_text == text:
            continue
        recovered = _recover_truncated_children_candidate(candidate_text)
        if recovered is not None:
            return recovered
    return None


def _is_empty_children_response(stripped: str) -> bool:
    fence_json = "\x60\x60\x60json"
    return not stripped or stripped in (fence_json, "error", '"""json"""') or fence_json in stripped


def _children_recovery_candidates(text: str) -> list[str]:
    candidates = [text]
    artifact_stripped = _strip_fenced_content_artifacts(text)
    if artifact_stripped != text:
        candidates.insert(0, artifact_stripped)
        sanitized_artifact = _sanitize_children_mapping_text_for_recovery(artifact_stripped)
        _insert_unique_candidate(candidates, sanitized_artifact, 1)
    sanitized_text = _sanitize_children_mapping_text_for_recovery(text)
    _append_unique_candidate(candidates, sanitized_text)
    extra_sanitized = _remove_nonstring_bracket_patterns(sanitized_text)
    _append_unique_candidate(candidates, extra_sanitized)
    comma_fixed = re.sub(r",\s*\}\s*\]", "]", sanitized_text)
    _append_unique_candidate(candidates, comma_fixed)
    return candidates


def _insert_unique_candidate(candidates: list[str], candidate: str, index: int) -> None:
    if candidate not in candidates:
        candidates.insert(index, candidate)


def _append_unique_candidate(candidates: list[str], candidate: str) -> None:
    if candidate not in candidates:
        candidates.append(candidate)


def _recover_children_candidate(text: str) -> dict[str, list[str]] | None:
    recovered = _recover_fenced_python_children_mapping(text)
    if recovered is not None:
        return recovered
    if text.count("{") <= 1 and text.count("}") <= 1:
        recovered = _recover_double_quoted_children_values(text)
        if recovered is not None:
            return recovered
    return _recover_standard_children_candidate(text)


def _recover_standard_children_candidate(text: str) -> dict[str, list[str]] | None:
    for recovery in (
        _recover_children_mapping_from_outer_block,
        _recover_malformed_children_mapping,
        _recover_inline_children_mapping,
        _recover_children_mapping_from_lines,
        _recover_unquoted_key_comma_separated,
    ):
        recovered = recovery(text)
        if recovered is not None:
            return recovered
    return None


def _recover_truncated_children_candidate(text: str) -> dict[str, list[str]] | None:
    for recovery in (
        _recover_fenced_python_children_mapping,
        _recover_inline_children_mapping,
    ):
        recovered = recovery(text)
        if recovered is not None:
            return recovered
    return None


def _recover_label_list_response(text: str) -> object | None:
    for recovery in (
        _recover_label_list_from_lines,
        _recover_bare_comma_separated_label_list,
        _recover_quoted_label_list_with_comments,
        _recover_single_bare_label,
    ):
        recovered = recovery(text)
        if recovered is not None:
            return recovered
    return None


def _recover_edge_list_response(text: str) -> object | None:
    for recovery in (
        _recover_bracketed_edge_pairs,
        _recover_bare_comma_separated_edge_pair,
    ):
        recovered = recovery(text)
        if recovered is not None:
            return recovered
    recovered = _recover_tuple_edge_pairs(text)
    if recovered is not None:
        return recovered
    return _extract_recoverable_edge_endpoints(text)


def _recover_tuple_edge_pairs(text: str) -> list[tuple[str, str]] | None:
    tuple_matches = re.findall(r"\(([^()]*)\)", text)
    parsed_edges: list[tuple[str, str]] = []
    for tuple_text in tuple_matches:
        edge = _parse_tuple_edge_pair(tuple_text)
        if edge is not None:
            parsed_edges.append(edge)
    return parsed_edges or None


def _parse_tuple_edge_pair(tuple_text: str) -> tuple[str, str] | None:
    parts = [part.strip().strip("'\"") for part in tuple_text.split(",", 1)]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return parts[0], parts[1]


def _recover_vote_list_response(text: str) -> list[str] | None:
    token_matches = re.findall(r"\b[YyNn]\b", text)
    return [token.upper() for token in token_matches] or None


# Constants from _compat module
_QWEN_CHAT_MODEL = "Qwen/Qwen3.5-9B"
