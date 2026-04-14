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
        if last_fence_pos > fenced.start():
            # Skip past the opening fence lang line (```lang\n)
            open_lang_end = text.find("\n", fenced.start())
            if open_lang_end < 0:
                open_lang_end = fenced.end()
            else:
                open_lang_end += 1  # include the newline
            body = text[open_lang_end:last_fence_pos].strip()
    return body


def _normalize_schema_response(parsed: object, *, schema_name: str) -> object:
    if isinstance(parsed, str):
        recovered = _recover_non_json_response(text=parsed, schema_name=schema_name)
        if recovered is not None:
            return recovered
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
    lowered = stripped.lower()
    if "<think>" in lowered or "" in lowered:
        return True
    if schema_name == "edge_list":
        if stripped.startswith("[") and not stripped.rstrip().endswith("]"):
            return True
        quoted_items = re.findall(r"""['"]([^'"]+)['"]""", stripped)
        if quoted_items and len(quoted_items) % 2 == 1:
            return True
    if schema_name == "children_by_label":
        if stripped.startswith("{") and not stripped.rstrip().endswith("}"):
            return True
    return False


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
    # Handle degenerate non-JSON outputs that contain no useful structure.
    stripped = text.strip().lower()
    if schema_name == "children_by_label":
        # Model returned bare markdown code fence (```json) or literal "Error".
        if not stripped or stripped in ("```json", "error", '"""json"""') or "```json" in stripped:
            return {"children_by_label": {}}
        candidate_texts = [text]
        # Strip thinking blocks, markdown bold markers, and embedded fences FIRST.
        # These are common model artifacts that corrupt JSON parsing before any recovery runs.
        artifact_stripped = _strip_fenced_content_artifacts(text)
        if artifact_stripped != text:
            candidate_texts.insert(0, artifact_stripped)
            # Also sanitize the artifact-stripped version
            sanitized_artifact = _sanitize_children_mapping_text_for_recovery(artifact_stripped)
            if (
                sanitized_artifact != artifact_stripped
                and sanitized_artifact not in candidate_texts
            ):
                candidate_texts.insert(1, sanitized_artifact)
        sanitized_text = _sanitize_children_mapping_text_for_recovery(text)
        if sanitized_text != text and sanitized_text not in candidate_texts:
            candidate_texts.append(sanitized_text)
        # Additional sanitization: remove non-string bracket patterns like [diet] reminders
        # or [exhaustion', low vitality'] that break the parser.
        extra_sanitized = _remove_nonstring_bracket_patterns(sanitized_text)
        if extra_sanitized != sanitized_text and extra_sanitized not in candidate_texts:
            candidate_texts.append(extra_sanitized)
        # Also handle trailing comma before } like ['Patience', 'Consistenc', }]
        comma_fixed = re.sub(r",\s*[\\]}]", "]", sanitized_text).rstrip()
        if comma_fixed != sanitized_text and comma_fixed not in candidate_texts:
            candidate_texts.append(comma_fixed)
        for candidate_text in candidate_texts:
            # Try fenced python children mapping FIRST (Mistral pattern with keys in quotes)
            # This needs to run before _remove_nonstring_bracket_patterns corrupts the text
            recovered_children = _recover_fenced_python_children_mapping(candidate_text)
            if recovered_children is not None:
                return {"children_by_label": recovered_children}
            if candidate_text.count("{") <= 1 and candidate_text.count("}") <= 1:
                recovered_children = _recover_double_quoted_children_values(candidate_text)
                if recovered_children is not None:
                    return {"children_by_label": recovered_children}
            recovered_children = _recover_children_mapping_from_outer_block(candidate_text)
            if recovered_children is not None:
                return {"children_by_label": recovered_children}
            recovered_children = _recover_malformed_children_mapping(candidate_text)
            if recovered_children is not None:
                return {"children_by_label": recovered_children}
            recovered_children = _recover_inline_children_mapping(candidate_text)
            if recovered_children is not None:
                return {"children_by_label": recovered_children}
            recovered_children = _recover_children_mapping_from_lines(candidate_text)
            if recovered_children is not None:
                return {"children_by_label": recovered_children}
            # Last resort: try unquoted key with comma-separated values
            recovered_children = _recover_unquoted_key_comma_separated(candidate_text)
            if recovered_children is not None:
                return {"children_by_label": recovered_children}
        # Also try truncated candidates for patterns like {Key: [val1, val2}
        truncated_candidates = _recover_truncated_children_mapping_blocks(text)
        for candidate_text in truncated_candidates:
            if candidate_text == text:
                continue  # already tried
            recovered_children = _recover_fenced_python_children_mapping(candidate_text)
            if recovered_children is not None:
                return {"children_by_label": recovered_children}
            recovered_children = _recover_inline_children_mapping(candidate_text)
            if recovered_children is not None:
                return {"children_by_label": recovered_children}
    if schema_name == "label_list":
        recovered_labels = _recover_label_list_from_lines(text)
        if recovered_labels is not None:
            return recovered_labels
        recovered_labels = _recover_bare_comma_separated_label_list(text)
        if recovered_labels is not None:
            return recovered_labels
        recovered_labels = _recover_quoted_label_list_with_comments(text)
        if recovered_labels is not None:
            return recovered_labels
        recovered_labels = _recover_single_bare_label(text)
        if recovered_labels is not None:
            return recovered_labels
    if schema_name == "edge_list":
        recovered_edge_pairs = _recover_bracketed_edge_pairs(text)
        if recovered_edge_pairs is not None:
            return recovered_edge_pairs
        recovered_edge_pairs = _recover_bare_comma_separated_edge_pair(text)
        if recovered_edge_pairs is not None:
            return recovered_edge_pairs
        tuple_matches = re.findall(r"\(([^()]*)\)", text)
        if tuple_matches:
            parsed_edges: list[tuple[str, str]] = []
            for tuple_text in tuple_matches:
                parts = [part.strip().strip("'\"") for part in tuple_text.split(",", 1)]
                if len(parts) != 2 or not parts[0] or not parts[1]:
                    continue
                parsed_edges.append((parts[0], parts[1]))
            if parsed_edges:
                return parsed_edges
        quoted_endpoints = _extract_recoverable_edge_endpoints(text)
        if quoted_endpoints is not None:
            return quoted_endpoints
    if schema_name == "vote_list":
        token_matches = re.findall(r"\b[YyNn]\b", text)
        if token_matches:
            return [token.upper() for token in token_matches]
    return None


# Constants from _compat module
_QWEN_CHAT_MODEL = "Qwen/Qwen3.5-9B"
