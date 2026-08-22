from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable, Mapping
from typing import TypeGuard

_MAPPING_CLOSING_BRACE = "}"


def _looks_like_children_mapping(parsed: object) -> TypeGuard[dict[str, list[str]]]:
    if not isinstance(parsed, dict) or "children_by_label" in parsed:
        return False
    return bool(parsed) and all(
        _is_children_mapping_entry(key, value) for key, value in parsed.items()
    )


def _is_children_mapping_entry(key: object, value: object) -> bool:
    return (
        isinstance(key, str)
        and isinstance(value, list)
        and all(isinstance(item, str) for item in value)
    )


def _recover_malformed_children_mapping(text: str) -> dict[str, list[str]] | None:
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    for candidate in _malformed_children_mapping_candidates(stripped):
        parsed = _try_literal_eval_children_mapping(candidate)
        normalized = _normalize_children_mapping_candidate(parsed)
        if normalized is not None:
            return normalized
    return None


def _malformed_children_mapping_candidates(stripped: str) -> list[str]:
    candidates = [f"{stripped}}}"]
    if stripped.endswith("]"):
        candidates.append(f"{stripped[:-1]}}}")
    return candidates


def _try_literal_eval_children_mapping(candidate: str) -> object | None:
    try:
        return ast.literal_eval(candidate)
    except (ValueError, SyntaxError):
        return None


def _recover_children_mapping_from_outer_block(text: str) -> dict[str, list[str]] | None:
    blocks_to_try = _children_mapping_blocks(text)
    for block in blocks_to_try:
        for candidate in _children_mapping_candidate_variants(block):
            parsed = _parse_children_mapping_candidate(candidate)
            if parsed is not None:
                return parsed
    return None


def _children_mapping_blocks(text: str) -> list[str]:
    from llm_conceptual_modeling.common.hf_transformers._label_list import (
        _extract_first_balanced_block,
        _extract_outer_block,
    )

    first_balanced = _extract_first_balanced_block(text, opener="{", closer="}")
    outer_block = _extract_outer_block(text=text, opener="{", closer="}")
    blocks: list[str] = []
    if first_balanced is not None:
        blocks.append(first_balanced)
    if outer_block is not None and outer_block != first_balanced:
        blocks.append(outer_block)
    return blocks


def _children_mapping_candidate_variants(block: str) -> tuple[str, ...]:
    stripped_comments = _strip_mapping_comments(block)
    if stripped_comments == block:
        return (block,)
    return block, stripped_comments


def _parse_children_mapping_candidate(candidate: str) -> dict[str, list[str]] | None:
    for parser in (json.loads, ast.literal_eval):
        parsed = _try_parse_children_mapping_candidate(candidate, parser)
        recovered = _normalize_children_mapping_candidate(parsed)
        if recovered is not None:
            return recovered
    return None


def _normalize_children_mapping_candidate(
    parsed: object | None,
) -> dict[str, list[str]] | None:
    if parsed is None:
        return None
    if isinstance(parsed, dict) and not parsed:
        return {}
    if _looks_like_children_mapping(parsed):
        return parsed
    return None


def _try_parse_children_mapping_candidate(
    candidate: str,
    parser: Callable[[str], object],
) -> object | None:
    try:
        return parser(candidate)
    except (json.JSONDecodeError, ValueError, SyntaxError):
        return None


def _strip_mapping_comments(text: str) -> str:
    without_block_comments = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    without_inline_comments = re.sub(r"//[^\n]*", "", without_block_comments)
    without_paren_notes = re.sub(
        r"\(\s*([^)]*)\)",
        _strip_mapping_parenthetical_note,
        without_inline_comments,
    )
    without_orphan_paren_lines = re.sub(
        r"(?m)^\s*\([^()]*\),?\s*$",
        "",
        without_paren_notes,
    )
    return _strip_mapping_hash_comments(without_orphan_paren_lines)


def _strip_mapping_parenthetical_note(match: re.Match[str]) -> str:
    content = match.group(1).strip()
    if not content:
        return match.group()
    first_word = content.split()[0].rstrip(":,").casefold()
    if first_word in {"note", "warning", "caveat", "correction", "actually"}:
        return ""
    return match.group()


def _strip_mapping_hash_comments(text: str) -> str:
    result: list[str] = []
    in_single_quote = False
    in_double_quote = False
    skip_until = 0
    for index, character in enumerate(text):
        if index < skip_until:
            continue
        skip_until, in_single_quote, in_double_quote = _consume_mapping_hash_character(
            text=text,
            index=index,
            character=character,
            result=result,
            in_single_quote=in_single_quote,
            in_double_quote=in_double_quote,
        )
    return "".join(result)


def _consume_mapping_hash_character(
    *,
    text: str,
    index: int,
    character: str,
    result: list[str],
    in_single_quote: bool,
    in_double_quote: bool,
) -> tuple[int, bool, bool]:
    if character == "\\":
        result.append(character)
        if index + 1 < len(text):
            result.append(text[index + 1])
        return index + 2, in_single_quote, in_double_quote
    if _is_mapping_quote(character, in_single_quote, in_double_quote):
        in_single_quote, in_double_quote = _toggle_mapping_quote(
            character,
            in_single_quote,
            in_double_quote,
        )
        result.append(character)
        return index + 1, in_single_quote, in_double_quote
    if _is_mapping_hash_comment_start(character, in_single_quote, in_double_quote):
        return _skip_mapping_hash_comment(text, index), in_single_quote, in_double_quote
    result.append(character)
    return index + 1, in_single_quote, in_double_quote


def _skip_mapping_hash_comment(text: str, index: int) -> int:
    newline_index = text.find("\n", index)
    if newline_index == -1:
        return len(text)
    return newline_index


def _is_mapping_quote(
    character: str,
    in_single_quote: bool,
    in_double_quote: bool,
) -> bool:
    return (character == "'" and in_double_quote is False) or (
        character == '"' and in_single_quote is False
    )


def _toggle_mapping_quote(
    character: str,
    in_single_quote: bool,
    in_double_quote: bool,
) -> tuple[bool, bool]:
    if character == "'":
        return not in_single_quote, in_double_quote
    return in_single_quote, not in_double_quote


def _is_mapping_hash_comment_start(
    character: str,
    in_single_quote: bool,
    in_double_quote: bool,
) -> bool:
    return character == "#" and not in_single_quote and not in_double_quote


def _recover_inline_children_mapping(text: str) -> dict[str, list[str]] | None:
    blocks_to_try = _inline_children_mapping_blocks(text)
    if not blocks_to_try:
        return None
    return _first_inline_children_mapping(blocks_to_try)


def _inline_children_mapping_blocks(text: str) -> list[str]:
    from llm_conceptual_modeling.common.hf_transformers._label_list import (
        _extract_first_balanced_block,
        _extract_outer_block,
    )

    first_balanced = _extract_first_balanced_block(text, opener="{", closer="}")
    outer_block = _extract_outer_block(text=text, opener="{", closer="}")
    blocks_to_try: list[str] = []
    if first_balanced is not None:
        blocks_to_try.append(first_balanced)
    if outer_block is not None and outer_block != first_balanced:
        blocks_to_try.append(outer_block)
    if not blocks_to_try:
        blocks_to_try.extend(_recover_truncated_children_mapping_blocks(text))
    return blocks_to_try


def _first_inline_children_mapping(
    blocks_to_try: list[str],
) -> dict[str, list[str]] | None:
    for block in blocks_to_try:
        result = _try_inline_children_parse(block)
        if result is not None:
            return result
    return None


def _try_inline_children_parse(block: str) -> dict[str, list[str]] | None:
    mapping: dict[str, list[str]] = {}
    index = 1
    length = len(block)
    while True:
        index = _skip_inline_mapping_separators(block, index)
        if _inline_children_mapping_finished(block, index, length):
            break
        next_index = _consume_inline_children_entry(mapping, block, index)
        if next_index is None:
            return None
        index = next_index
    if not mapping:
        return None
    return mapping


def _inline_children_mapping_finished(block: str, index: int, length: int) -> bool:
    return index >= length or block[index] == "}"


def _consume_inline_children_entry(
    mapping: dict[str, list[str]],
    block: str,
    index: int,
) -> int | None:
    entry = _parse_inline_children_entry(block, index)
    if entry is None:
        return None
    key, values, next_index = entry
    _store_inline_children_entry(mapping, key, values)
    return _skip_inline_mapping_comma(block, next_index)


def _parse_inline_children_entry(
    block: str,
    start_index: int,
) -> tuple[str, list[str], int] | None:
    key_result = _scan_lenient_mapping_key(block, start_index)
    if key_result is None:
        return None
    key, index = key_result
    index = _skip_inline_mapping_separators(block, index)
    index = _consume_inline_mapping_token(block, index, ":")
    if index is None:
        return None
    index = _skip_inline_mapping_separators(block, index)
    index = _consume_inline_mapping_token(block, index, "[")
    if index is None:
        return None
    values_result = _scan_lenient_quoted_list(block, index - 1)
    if values_result is None:
        return None
    values, index = values_result
    return key, values, index


def _consume_inline_mapping_token(
    text: str,
    index: int,
    token: str,
) -> int | None:
    if index >= len(text) or text[index] != token:
        return None
    return index + 1


def _store_inline_children_entry(
    mapping: dict[str, list[str]],
    key: str,
    values: list[str],
) -> None:
    existing_values = mapping.get(key)
    if existing_values and not values:
        return
    mapping[key] = values


def _skip_inline_mapping_comma(text: str, index: int) -> int:
    if index < len(text) and text[index] == ",":
        return index + 1
    return index


def _skip_inline_mapping_separators(text: str, index: int) -> int:
    separator_text = text[index:]
    return index + len(separator_text) - len(separator_text.lstrip(" \t\n\r"))


def _scan_lenient_mapping_key(text: str, start_index: int) -> tuple[str, int] | None:
    from llm_conceptual_modeling.common.hf_transformers._label_list import (
        _scan_lenient_quoted_string,
    )

    quoted_result = _scan_lenient_quoted_string(text, start_index)
    if quoted_result is not None:
        return quoted_result
    return _scan_unquoted_mapping_key(text, start_index)


def _scan_lenient_quoted_list(text: str, start_index: int) -> tuple[list[str], int] | None:
    from llm_conceptual_modeling.common.hf_transformers._label_list import (
        _scan_lenient_quoted_string,
    )

    if text[start_index] != "[":
        return None
    values: list[str] = []
    return _scan_lenient_list_items(text, start_index + 1, values, _scan_lenient_quoted_string)


def _scan_lenient_list_items(
    text: str,
    index: int,
    values: list[str],
    scan_quoted_string: Callable[[str, int], tuple[str, int] | None],
) -> tuple[list[str], int] | None:
    while True:
        index = _skip_inline_mapping_separators(text, index)
        if index >= len(text):
            return None
        step = _scan_lenient_list_position(text, index, values, scan_quoted_string)
        if step is None:
            return None
        if isinstance(step, tuple):
            return step
        index = step
    return None


def _scan_lenient_list_position(
    text: str,
    index: int,
    values: list[str],
    scan_quoted_string: Callable[[str, int], tuple[str, int] | None],
) -> tuple[list[str], int] | int | None:
    if text[index] in {"]", ")"}:
        return values, index + 1
    if _is_terminal_quoted_list_item(text, index):
        next_index = _skip_inline_mapping_separators(text, index + 1)
        return values, next_index + 1
    if text[index] == "[":
        return _scan_nested_list_position(text, index, values)
    return _scan_regular_lenient_list_item(text, index, values, scan_quoted_string)


def _scan_regular_lenient_list_item(
    text: str,
    index: int,
    values: list[str],
    scan_quoted_string: Callable[[str, int], tuple[str, int] | None],
) -> tuple[list[str], int] | int | None:
    item_result = scan_quoted_string(text, index) or _scan_unquoted_list_item(text, index)
    if item_result is None:
        return None
    item, item_end = item_result
    values.append(item)
    return _advance_after_lenient_list_item(text, item_end, values)


def _is_terminal_quoted_list_item(text: str, index: int) -> bool:
    if text[index] not in {"'", '"'}:
        return False
    next_index = _skip_inline_mapping_separators(text, index + 1)
    return next_index < len(text) and text[next_index] in {"]", "}", ")"}


def _scan_nested_list_position(
    text: str,
    index: int,
    values: list[str],
) -> tuple[list[str], int] | int | None:
    next_index = _skip_nested_bracketed_value(text, index)
    if next_index == -1:
        return None
    next_index = _skip_inline_mapping_separators(text, next_index)
    return _finish_nested_list_position(text, next_index, values)


def _finish_nested_list_position(
    text: str,
    index: int,
    values: list[str],
) -> tuple[list[str], int] | int | None:
    if index < len(text) and text[index] == ",":
        return index + 1
    if index < len(text) and text[index] in {"]", ")"}:
        return values, index + 1
    return None


def _advance_after_lenient_list_item(
    text: str,
    index: int,
    values: list[str],
) -> tuple[list[str], int] | int | None:
    next_index = _skip_inline_mapping_separators(text, index)
    if next_index < len(text) and text[next_index] == ",":
        return next_index + 1
    if next_index < len(text) and text[next_index] in {"]", ")"}:
        return values, next_index + 1
    return None


def _skip_nested_bracketed_value(text: str, start_index: int) -> int:
    if start_index >= len(text) or text[start_index] != "[":
        return -1
    depth = 0
    for index in range(start_index, len(text)):
        depth, closed = _advance_nested_bracket_depth(text[index], depth)
        if closed:
            return index + 1
    return -1


def _advance_nested_bracket_depth(character: str, depth: int) -> tuple[int, bool]:
    if character == "[":
        return depth + 1, False
    if character == "]":
        depth -= 1
        return depth, depth == 0
    return depth, False


def _scan_unquoted_list_item(text: str, start_index: int) -> tuple[str, int] | None:
    """Scan a list item that lacks an opening quote."""
    delimiter_chars = {",", "]", "}", ")"}
    for index in range(start_index, len(text)):
        current_char = text[index]
        if current_char in delimiter_chars:
            return _finish_unquoted_list_item(text, start_index, index)
        if current_char in {"'", '"'}:
            if _is_terminal_unquoted_quote(text, index, delimiter_chars):
                return _finish_unquoted_list_item(text, start_index, index, index + 1)
    return None


def _finish_unquoted_list_item(
    text: str,
    start_index: int,
    end_index: int,
    result_index: int | None = None,
) -> tuple[str, int] | None:
    value = text[start_index:end_index].strip()
    if not value:
        return None
    return value, end_index if result_index is None else result_index


def _is_terminal_unquoted_quote(
    text: str,
    index: int,
    delimiter_chars: set[str],
) -> bool:
    next_index = _skip_inline_mapping_separators(text, index + 1)
    return next_index >= len(text) or text[next_index] in delimiter_chars


def _scan_unquoted_mapping_key(
    text: str,
    start_index: int | None,
) -> tuple[str, int] | None:
    if start_index is None:
        return None
    colon_index = text.find(":", start_index)
    if colon_index == -1:
        return None
    value = _clean_unquoted_mapping_key(text[start_index:colon_index])
    if value is None:
        return None
    return value, colon_index


def _clean_unquoted_mapping_key(key_text: str) -> str | None:
    if any(character in {"}", "]", ","} for character in key_text):
        return None
    value = key_text.strip().strip("{").strip()
    if not value:
        return None
    return value.strip("'\"").strip()


def _sanitize_children_mapping_text_for_recovery(text: str) -> str:
    sanitized = text.strip()
    if sanitized.startswith('"') and not sanitized.endswith('"'):
        sanitized = sanitized[1:]
    if sanitized.endswith("\\"):
        sanitized = sanitized[:-1]
    # Fix Mistral artifact where outer quote is swapped with inner quote at item end
    # e.g., '"Thinspiration"', or '"Thinspiration"\', -> '"Thinspiration"' ,
    sanitized = re.sub(r"\\?'\"\s*([,\]\n])", '"' + "'" + r"\1", sanitized)
    sanitized = re.sub(
        r"\\u([0-9a-fA-F]{4})",
        lambda match: chr(int(match.group(1), 16)),
        sanitized,
    )
    replacements = {
        chr(0x2018): "'",
        chr(0x2019): "'",
        chr(0x201C): '"',
        chr(0x201D): '"',
        chr(0xFF1A): ":",
    }
    for source, target in replacements.items():
        sanitized = sanitized.replace(source, target)
    return sanitized


def _strip_fenced_content_artifacts(text: str) -> str:
    """Strip model-generated artifacts from fenced code block content.

    Handles three common artifact families in Mistral/Qwen contrastive outputs:
    1. Embedded ``` inside JSON strings (e.g., '"Key\n```ng"') - strips the embedded fence
    2. <think>...</think> thinking block content - removes thinking text AND delimiters
    3. **bold** markdown markers - strips bold wrappers, keeps inner text

    These artifacts break JSON parsing and are not valid JSON structure.
    """
    # 1. Strip embedded fences inside strings: \n```lang (at end of a string value)
    result = re.sub(r"\n```[a-zA-z0-9]*", "", text)
    # 2. Strip <think>...</think> thinking blocks (content AND delimiters)
    result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL)
    # 3. Strip **bold** markdown markers, keep inner text
    result = re.sub(r"\*\*(.+?)\*\*", r"\1", result)
    # 4. After bold strip, trailing '** :' or '**:' may remain at end of line.
    #    Strip trailing '** :' and '**:' with their trailing quote/colons cleanly.
    #    Replace '** :' with ':' and '**:' with ':' at end of content lines.
    result = re.sub(r"\*\*\s*:\s*", ":", result)  # '** :' -> ':'
    result = re.sub(r"\*\*", "", result)  # remove remaining '**'
    return result


def _recover_unquoted_key_comma_separated(text: str) -> dict[str, list[str]] | None:
    """Recover from unquoted key with comma-separated values (no brackets).

    Handles: {KeyName: Value1, Value2, Value3}
    Returns: {"KeyName": ["Value1", "Value2", "Value3"]}
    """
    stripped = text.strip()
    if not _is_unquoted_children_mapping_candidate(stripped):
        return None
    key_and_values = _split_unquoted_children_mapping(stripped)
    if key_and_values is None:
        return None
    key, values_part = key_and_values
    values = _parse_unquoted_children_values(values_part)
    if not key or not values:
        return None
    return {key: values}


def _is_unquoted_children_mapping_candidate(text: str) -> bool:
    # Bracketed payloads are handled by other recovery functions.
    return text.startswith("{") and "[" not in text and "]" not in text


def _split_unquoted_children_mapping(text: str) -> tuple[str, str] | None:
    colon_pos = text.find(":")
    if colon_pos == -1:
        return None
    return text[1:colon_pos].strip(), text[colon_pos + 1 :].strip().rstrip("}")


def _parse_unquoted_children_values(values_part: str) -> list[str]:
    items = re.split(r",\s*", values_part)
    return [value.strip().strip("'\"").strip() for value in items if value.strip()]


def _recover_fenced_python_children_mapping(text: str) -> dict[str, list[str]] | None:
    """Recover from fenced python block or inline python with children mapping missing colons.

    Handles:
    ```python
    {
        "anorexia nervosa" ["starvation behavior", ...],
        "bulimiaproblematique" ["binge-eatingepisodestendency"]
    }
    ```
    Also handles text already stripped of fence markers.
    Returns: {"anorexia nervosa": ["starvation behavior", ...], ...}
    """
    block = _extract_fenced_children_block(text)
    if block is None:
        return None
    for recovery in (
        _recover_structured_fenced_children_mapping,
        _recover_fenced_children_mapping_from_lines,
        _recover_fenced_children_fallback,
    ):
        recovered = recovery(block)
        if recovered is not None:
            return recovered
    return None


def _extract_fenced_children_block(text: str) -> str | None:
    stripped = text.strip()
    fence_marker = chr(96) * 3
    match = re.search(
        rf"{re.escape(fence_marker)}(?:\w+)?\s*(.*?){re.escape(fence_marker)}",
        stripped,
        flags=re.DOTALL,
    )
    if match is None:
        block = _strip_truncated_fence(stripped)
        if block.endswith(fence_marker):
            return None
    else:
        block = _fenced_match_body(match)
    if not block.startswith("{"):
        return None
    block = _strip_fenced_content_artifacts(block)
    return _close_truncated_children_mapping(block)


def _fenced_match_body(match: re.Match[str]) -> str:
    return match.group(1).strip()


def _strip_truncated_fence(text: str) -> str:
    fence_marker = chr(96) * 3
    if not text.startswith(fence_marker):
        return text
    return re.sub(r"^\x60\x60\x60(?:\w+)?\s*", "", text).strip()


def _close_truncated_children_mapping(block: str) -> str:
    normalized = block.rstrip()
    if normalized.endswith("]") and not normalized.endswith("}]"):
        return normalized + "}"
    return normalized


def _recover_structured_fenced_children_mapping(
    block: str,
) -> dict[str, list[str]] | None:
    quote_closed_candidate = re.sub(r"\s*:\s*\n\s*\[", "', ", block)
    if quote_closed_candidate != block:
        quoted_entry = _recover_first_quoted_children_entry(quote_closed_candidate)
        if quoted_entry is not None:
            return quoted_entry
    for candidate in (block, _strip_mapping_comments(block)):
        parsed = _try_recovered_fenced_children_mapping(candidate)
        if parsed is not None:
            return parsed
    return None


def _try_recovered_fenced_children_mapping(
    block: str,
) -> dict[str, list[str]] | None:
    parsed = _try_inline_children_parse(block)
    if parsed is None or _children_mapping_needs_more_recovery(text=block, parsed=parsed):
        return None
    return parsed


def _recover_fenced_children_mapping_from_lines(
    block: str,
) -> dict[str, list[str]] | None:
    joined_block = re.sub(r"\n\s+", " ", block)
    mapping: dict[str, list[str]] = {}
    for raw_line in joined_block.splitlines():
        parsed_line = _parse_fenced_children_line(raw_line)
        if parsed_line is not None:
            key, values = parsed_line
            mapping[key] = values
    return mapping or None


def _recover_fenced_children_fallback(
    block: str,
) -> dict[str, list[str]] | None:
    first_entry = _recover_first_quoted_children_entry(block)
    if first_entry is None:
        return None
    if _children_mapping_needs_more_recovery(text=block, parsed=first_entry):
        return None
    return first_entry


def _parse_fenced_children_line(line: str) -> tuple[str, list[str]] | None:
    normalized_line = _normalize_fenced_children_line(line)
    if normalized_line is None:
        return None
    return _match_fenced_children_line(normalized_line)


def _normalize_fenced_children_line(line: str) -> str | None:
    normalized_line = line.strip().rstrip(",")
    normalized_line = re.sub(r"^[\s{,\\n]+", "", normalized_line)
    normalized_line = re.sub(r"[\s,]*(}[,\s]*)$", r"\1", normalized_line)
    if not normalized_line or normalized_line == _MAPPING_CLOSING_BRACE:
        return None
    return normalized_line


def _fenced_children_line_patterns() -> tuple[tuple[str, bool], ...]:
    return (
        (r""""([^"]+)"(?:\s*\([^)]*\))?(?!\s*:)\s*\[([^\]]+)\]""", False),
        (r"""'([^']+)'(?:\s*\([^)]*\))?(?!\s*:)\s*\[([^\]]+)\]""", False),
        (r""""([^"]+)"(?:\s*\([^)]*\)):\s*\[([^\]]+)\]""", False),
        (r"""'([^']+)'(?:\s*\([^)]*\)):\s*\[([^\]]+)\]""", False),
        (
            '"' + "'" + "([^" + "'" + "]+)" + "'" + r"(?:\s*\([^)]*\)):\s*\[([^\]]+)\]",
            True,
        ),
    )


def _match_fenced_children_line(line: str) -> tuple[str, list[str]] | None:
    for pattern, wrap_key in _fenced_children_line_patterns():
        match = re.match(pattern, line)
        if match is None:
            continue
        key = match.group(1)
        if wrap_key:
            key = f"'{key}'"
        values = _split_fenced_children_values(match.group(2))
        if values:
            return key, values
    return None


def _split_fenced_children_values(values_text: str) -> list[str]:
    values = [value.strip().strip("'\"") for value in values_text.split(",")]
    return [value for value in values if value]


def _recover_first_quoted_children_entry(text: str) -> dict[str, list[str]] | None:
    patterns = (
        r"""'([^']+)'\s*:\s*\[(.*?)\]""",
        r""""([^"]+)"\s*:\s*\[(.*?)\]""",
    )
    for pattern in patterns:
        recovered = _recover_first_quoted_children_entry_with_pattern(text, pattern)
        if recovered is not None:
            return recovered
    return None


def _recover_first_quoted_children_entry_with_pattern(
    text: str,
    pattern: str,
) -> dict[str, list[str]] | None:
    match = re.search(pattern, text, flags=re.DOTALL)
    if match is None:
        return None
    key = match.group(1).strip()
    values_result = _scan_lenient_quoted_list(f"[{match.group(2)}]", 0)
    if values_result is None:
        return None
    values, _ = values_result
    if not key or not values:
        return None
    return {key: values}


def _recover_single_key_children_values(text: str) -> dict[str, list[str]] | None:
    match = re.search(r"""['"]([^'"]+)['"]\s*:\s*\[""", text)
    if match is None:
        return None
    key = match.group(1).strip()
    values_text = text[match.end() :]
    return _build_single_key_children_values(key, values_text)


def _build_single_key_children_values(
    key: str,
    values_text: str,
) -> dict[str, list[str]] | None:
    if not key:
        return None
    values = _extract_nonempty_quoted_values(values_text)
    if not values:
        return None
    return {key: values}


def _extract_nonempty_quoted_values(text: str) -> list[str]:
    return [item.strip() for item in re.findall(r"""['"]([^'"]+)['"]""", text) if item.strip()]


def _recover_double_quoted_children_values(text: str) -> dict[str, list[str]] | None:
    match = re.search(r'"([^"]+)"\s*:\s*\[', text)
    if match is None:
        return None
    key = match.group(1).strip()
    if not key:
        return None
    values = [item.strip() for item in re.findall(r'"([^"]+)"', text[match.end() :])]
    if not values:
        return None
    return {key: values}


def _children_mapping_needs_more_recovery(
    *,
    text: str,
    parsed: Mapping[str, list[str]],
) -> bool:
    if len(parsed) != 1:
        return False
    values = next(iter(parsed.values()))
    return _children_values_need_more_recovery(values) or _children_text_needs_more_recovery(text)


def _children_values_need_more_recovery(values: list[str]) -> bool:
    return any(_child_value_needs_more_recovery(value) for value in values)


def _child_value_needs_more_recovery(value: str) -> bool:
    return (
        value.startswith(('"', "'"))
        or value.endswith(('"', "'"))
        or "[" in value
        or "]" in value
        or "\n" in value
    )


def _children_text_needs_more_recovery(text: str) -> bool:
    return bool(re.search(r"\[[^\]\"'\[]+\]", text))


def _remove_nonstring_bracket_patterns(text: str) -> str:
    """Remove non-string bracket patterns that break children_by_label recovery.

    Handles cases like:
    - '[diet] reminders'  -> removes [diet] and trailing garbage
    - "[exhaustion', low vitality']"  -> extracts valid strings
    """
    result = text
    # Remove bare [word] followed by space and more content
    result = re.sub(r"\[[^\"\[]+\][^\[\"]*", "", result)
    # Handle nested bracket patterns - extract quoted strings from within
    return re.sub(
        r"\[([^'\"]*'[^'\"]*'[^'\"]*)'?\s*\]",
        lambda m: _extract_quoted_strings_from_bracket(m.group(1)),
        result,
    )


def _extract_quoted_strings_from_bracket(content: str) -> str:
    """Extract quoted strings from bracket content, return them comma-separated."""
    quoted = re.findall(r"""['"]([^'"]+)['"]""", content)
    if quoted:
        return ", ".join(f'"{s}"' for s in quoted if s.strip())
    return ""


def _recover_truncated_children_mapping_blocks(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped.startswith("{"):
        return []
    candidates = [stripped, *_truncated_children_mapping_delimiter_candidates(stripped)]
    trailing_comma_fixed = re.sub(r",\s*\}\s*\]", "]", stripped)
    if trailing_comma_fixed != stripped:
        candidates.append(trailing_comma_fixed)
    return _deduplicate_text_candidates(candidates)


def _truncated_children_mapping_delimiter_candidates(stripped: str) -> list[str]:
    open_list = stripped.count("[") > stripped.count("]")
    open_mapping = stripped.count("{") > stripped.count("}")
    candidates: list[str] = []
    _append_truncated_candidate(candidates, open_list, f"{stripped}]")
    _append_truncated_candidate(candidates, open_mapping, f"{stripped}}}")
    _append_truncated_candidate(candidates, open_list and open_mapping, f"{stripped}]}}")
    _append_truncated_candidate(
        candidates,
        stripped.endswith("}") and open_list,
        stripped[:-1] + "]}}",
    )
    return candidates


def _append_truncated_candidate(
    candidates: list[str],
    condition: bool,
    candidate: str,
) -> None:
    if condition:
        candidates.append(candidate)


def _deduplicate_text_candidates(candidates: list[str]) -> list[str]:
    deduplicated: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduplicated.append(candidate)
    return deduplicated


def _recover_children_mapping_from_lines(text: str) -> dict[str, list[str]] | None:
    from llm_conceptual_modeling.common.hf_transformers._label_list import (
        _extract_outer_block,
        _extract_quoted_line_value,
    )

    prepared = _prepare_children_mapping_lines(text, _extract_outer_block)
    if prepared is None:
        return None
    sanitized_text, lines = prepared
    mapping: dict[str, list[str]] = {}
    key: str | None = None
    values: list[str] = []
    in_children = False

    for line in lines:
        key, values, in_children = _consume_children_mapping_line(
            line=line,
            mapping=mapping,
            key=key,
            values=values,
            in_children=in_children,
            extract_quoted_line_value=_extract_quoted_line_value,
        )
    key, values, _ = _flush_children_mapping_line_entry(mapping, key, values)
    return _finish_children_mapping_recovery(sanitized_text, mapping)


def _prepare_children_mapping_lines(
    text: str,
    extract_outer_block: Callable[..., str | None],
) -> tuple[str, list[str]] | None:
    block = extract_outer_block(text=text, opener="{", closer="}")
    candidate_text = block if block is not None else text
    sanitized_text = _strip_mapping_comments(candidate_text)
    lines = [line.strip() for line in sanitized_text.splitlines() if line.strip()]
    if not lines:
        return None
    return sanitized_text, lines


def _finish_children_mapping_recovery(
    sanitized_text: str,
    mapping: dict[str, list[str]],
) -> dict[str, list[str]] | None:
    if not mapping:
        return None
    if _children_mapping_needs_more_recovery(text=sanitized_text, parsed=mapping):
        return None
    return mapping


def _consume_children_mapping_line(
    *,
    line: str,
    mapping: dict[str, list[str]],
    key: str | None,
    values: list[str],
    in_children: bool,
    extract_quoted_line_value: Callable[[str], str | None],
) -> tuple[str | None, list[str], bool]:
    if line in {"{", _MAPPING_CLOSING_BRACE}:
        if line == _MAPPING_CLOSING_BRACE:
            return _flush_children_mapping_line_entry(mapping, key, values)
        return key, values, in_children
    if ":" in line:
        return _consume_children_mapping_assignment(
            line=line,
            mapping=mapping,
            key=key,
            values=values,
            in_children=in_children,
            extract_quoted_line_value=extract_quoted_line_value,
        )
    return _consume_children_mapping_value(
        line=line,
        mapping=mapping,
        key=key,
        values=values,
        in_children=in_children,
        extract_quoted_line_value=extract_quoted_line_value,
    )


def _consume_children_mapping_assignment(
    *,
    line: str,
    mapping: dict[str, list[str]],
    key: str | None,
    values: list[str],
    in_children: bool,
    extract_quoted_line_value: Callable[[str], str | None],
) -> tuple[str | None, list[str], bool]:
    line_key, line_value = line.split(":", 1)
    candidate_key = extract_quoted_line_value(line_key)
    if candidate_key is None:
        return _consume_children_mapping_value(
            line=line,
            mapping=mapping,
            key=key,
            values=values,
            in_children=in_children,
            extract_quoted_line_value=extract_quoted_line_value,
        )
    key, values, in_children = _flush_children_mapping_line_entry(mapping, key, values)
    key = candidate_key
    if "[" not in line_value:
        return key, values, in_children
    in_children = True
    first_value_text = line_value.split("[", 1)[1]
    value = extract_quoted_line_value(first_value_text)
    if value is not None:
        values.append(value)
    if "]" in first_value_text:
        return _flush_children_mapping_line_entry(mapping, key, values)
    return key, values, in_children


def _consume_children_mapping_value(
    *,
    line: str,
    mapping: dict[str, list[str]],
    key: str | None,
    values: list[str],
    in_children: bool,
    extract_quoted_line_value: Callable[[str], str | None],
) -> tuple[str | None, list[str], bool]:
    if key is None or not in_children:
        return key, values, in_children
    if line.startswith("]") or line.startswith(_MAPPING_CLOSING_BRACE):
        return _flush_children_mapping_line_entry(mapping, key, values)
    return _append_children_mapping_line_value(
        line=line,
        key=key,
        values=values,
        in_children=in_children,
        extract_quoted_line_value=extract_quoted_line_value,
    )


def _append_children_mapping_line_value(
    *,
    line: str,
    key: str,
    values: list[str],
    in_children: bool,
    extract_quoted_line_value: Callable[[str], str | None],
) -> tuple[str, list[str], bool]:
    value = extract_quoted_line_value(line)
    if value is not None:
        values.append(value)
    return key, values, in_children


def _flush_children_mapping_line_entry(
    mapping: dict[str, list[str]],
    key: str | None,
    values: list[str],
) -> tuple[str | None, list[str], bool]:
    if key is not None:
        mapping[key] = values
    return None, [], False
