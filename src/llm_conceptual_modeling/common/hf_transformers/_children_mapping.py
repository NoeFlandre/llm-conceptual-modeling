from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping


def _looks_like_children_mapping(parsed: object) -> bool:
    if not isinstance(parsed, dict) or "children_by_label" in parsed:
        return False
    if not parsed:
        return False
    for key, value in parsed.items():
        if not isinstance(key, str):
            return False
        if not isinstance(value, list):
            return False
        if not all(isinstance(item, str) for item in value):
            return False
    return True


def _recover_malformed_children_mapping(text: str) -> dict[str, list[str]] | None:
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    candidates = [f"{stripped}}}"]
    if stripped.endswith("]"):
        candidates.append(f"{stripped[:-1]}}}")
    for candidate in candidates:
        try:
            parsed = ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            continue
        if _looks_like_children_mapping(parsed):
            return parsed
    return None


def _recover_children_mapping_from_outer_block(text: str) -> dict[str, list[str]] | None:
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
        return None
    for block in blocks_to_try:
        candidates = [block]
        stripped_comments = _strip_mapping_comments(block)
        if stripped_comments != block:
            candidates.append(stripped_comments)
        for candidate in candidates:
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(candidate)
                except (json.JSONDecodeError, ValueError, SyntaxError):
                    continue
                if isinstance(parsed, dict) and not parsed:
                    return {}
                if _looks_like_children_mapping(parsed):
                    return parsed
    return None


def _strip_mapping_comments(text: str) -> str:
    without_block_comments = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    without_inline_comments = re.sub(r"//[^\n]*", "", without_block_comments)
    without_paren_notes = re.sub(
        r"\(\s*(?:Note|WARNING|Caveat|Correction|Actually)\b[^)]*\)",
        "",
        without_inline_comments,
        flags=re.IGNORECASE,
    )
    without_orphan_paren_lines = re.sub(
        r"(?m)^\s*\([^()]*\),?\s*$",
        "",
        without_paren_notes,
    )
    result: list[str] = []
    index = 0
    in_single_quote = False
    in_double_quote = False
    while index < len(without_orphan_paren_lines):
        character = without_orphan_paren_lines[index]
        if character == "\\":
            result.append(character)
            index += 1
            if index < len(without_orphan_paren_lines):
                result.append(without_orphan_paren_lines[index])
            index += 1
            continue
        if character == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            result.append(character)
            index += 1
            continue
        if character == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            result.append(character)
            index += 1
            continue
        if character == "#" and not in_single_quote and not in_double_quote:
            while (
                index < len(without_orphan_paren_lines)
                and without_orphan_paren_lines[index] != "\n"
            ):
                index += 1
            continue
        result.append(character)
        index += 1
    return "".join(result)


def _recover_inline_children_mapping(text: str) -> dict[str, list[str]] | None:
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
    if not blocks_to_try:
        return None
    for block in blocks_to_try:
        result = _try_inline_children_parse(block)
        if result is not None:
            return result
    return None


def _try_inline_children_parse(block: str) -> dict[str, list[str]] | None:
    mapping: dict[str, list[str]] = {}
    index = 1
    length = len(block)
    while index < length:
        index = _skip_inline_mapping_separators(block, index)
        if index >= length or block[index] == "}":
            break
        key_result = _scan_lenient_mapping_key(block, index)
        if key_result is None:
            return None
        key, index = key_result
        index = _skip_inline_mapping_separators(block, index)
        if index >= length or block[index] != ":":
            return None
        index += 1
        index = _skip_inline_mapping_separators(block, index)
        if index >= length or block[index] != "[":
            return None
        values_result = _scan_lenient_quoted_list(block, index)
        if values_result is None:
            return None
        values, index = values_result
        existing_values = mapping.get(key)
        if existing_values and not values:
            pass
        else:
            mapping[key] = values
        index = _skip_inline_mapping_separators(block, index)
        if index < length and block[index] == ",":
            index += 1
    if not mapping:
        return None
    return mapping


def _skip_inline_mapping_separators(text: str, index: int) -> int:
    while index < len(text) and text[index] in {" ", "\t", "\n", "\r"}:
        index += 1
    return index


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
    index = start_index + 1
    values: list[str] = []
    while index < len(text):
        index = _skip_inline_mapping_separators(text, index)
        if index >= len(text):
            return None
        if text[index] in {"]", ")"}:
            return values, index + 1
        if text[index] in {"'", '"'}:
            next_index = _skip_inline_mapping_separators(text, index + 1)
            if next_index < len(text) and text[next_index] in {"]", "}", ")"}:
                return values, next_index + 1
        if text[index] == "[":
            index = _skip_nested_bracketed_value(text, index)
            if index == -1:
                return None
            index = _skip_inline_mapping_separators(text, index)
            if index < len(text) and text[index] == ",":
                index += 1
                continue
            if index < len(text) and text[index] in {"]", ")"}:
                return values, index + 1
            return None
        item_result = _scan_lenient_quoted_string(text, index)
        if item_result is None:
            item_result = _scan_unquoted_list_item(text, index)
        if item_result is None:
            return None
        item, index = item_result
        values.append(item)
        index = _skip_inline_mapping_separators(text, index)
        if index < len(text) and text[index] == ",":
            index += 1
            continue
        if index < len(text) and text[index] in {"]", ")"}:
            return values, index + 1
        return None
    return None


def _skip_nested_bracketed_value(text: str, start_index: int) -> int:
    if start_index >= len(text) or text[start_index] != "[":
        return -1
    depth = 0
    index = start_index
    while index < len(text):
        character = text[index]
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return -1


def _scan_unquoted_list_item(text: str, start_index: int) -> tuple[str, int] | None:
    """Scan a list item that lacks an opening quote."""
    delimiter_chars = {",", "]", "}", ")"}
    index = start_index
    while index < len(text):
        current_char = text[index]
        if current_char in delimiter_chars:
            value = text[start_index:index].strip()
            if value:
                return value, index
            return None
        if current_char in {"'", '"'}:
            next_index = _skip_inline_mapping_separators(text, index + 1)
            if next_index >= len(text) or text[next_index] in delimiter_chars:
                value = text[start_index:index].strip()
                if value:
                    return value, index + 1
                return None
        index += 1
    return None


def _scan_unquoted_mapping_key(text: str, start_index: int) -> tuple[str, int] | None:
    index = start_index
    while index < len(text) and text[index] != ":":
        if text[index] in {"}", "]", ","}:
            return None
        index += 1
    if index >= len(text) or text[index] != ":":
        return None
    value = text[start_index:index].strip().strip("{").strip()
    if not value:
        return None
    value = value.strip("'\"").strip()
    return value, index


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
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\uff1a": ":",
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
    if not stripped.startswith("{"):
        return None
    # Skip texts that have brackets - those are handled by other recovery functions
    if "[" in stripped or "]" in stripped:
        return None

    # Find the first colon that separates key from values
    colon_pos = stripped.find(":")
    if colon_pos == -1:
        return None

    key = stripped[1:colon_pos].strip()
    values_part = stripped[colon_pos + 1 :].strip().rstrip("}")

    # Split on commas to get individual values
    items = re.split(r",\s*", values_part)
    values = [v.strip().strip("'\"").strip() for v in items if v.strip()]

    if not key or not values:
        return None

    return {key: values}


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
    stripped = text.strip()
    block = stripped

    # Try to extract fenced python block content
    match = re.search(r"```(?:\w+)?\s*(.*?)```", stripped, flags=re.DOTALL)
    if match:
        block = match.group(1).strip()
    elif stripped.startswith("```"):
        # Truncated fence - strip opening ``` and try to parse content
        block = re.sub(r"^```(?:\w+)?\s*", "", stripped).strip()

    if not block.startswith("{"):
        return None

    block = _strip_fenced_content_artifacts(block)

    # Handle truncated dict: ends with ] but missing closing }
    block = block.rstrip()
    if block.endswith("]") and not block.endswith("}]"):
        block = block + "}"

    quote_closed_candidate = re.sub(r"\s*:\s*\n\s*\[", "', ", block)
    if quote_closed_candidate != block:
        quoted_entry = _recover_first_quoted_children_entry(quote_closed_candidate)
        if quoted_entry is not None:
            return quoted_entry

    parsed_block = _try_inline_children_parse(block)
    if parsed_block is not None and not _children_mapping_needs_more_recovery(
        text=block, parsed=parsed_block
    ):
        return parsed_block

    block = _strip_mapping_comments(block)
    parsed_block = _try_inline_children_parse(block)
    if parsed_block is not None and not _children_mapping_needs_more_recovery(
        text=block, parsed=parsed_block
    ):
        return parsed_block

    # Join continuation lines (multi-line values) before parsing
    block = re.sub(r"\n\s+", " ", block)

    mapping: dict[str, list[str]] = {}
    for line in block.splitlines():
        line = line.strip().rstrip(",")
        # Strip leading dict artifacts but preserve key pattern
        line = re.sub(r"^[\s{,\\n]+", "", line)
        # Strip trailing dict artifacts but preserve ] at the end
        line = re.sub(r"[\s,]*(}[,\s]*)$", r"\1", line)
        if not line or line in {"}", "},"}:
            continue

        # Try fenced python patterns for Mistral children mapping without colons
        # These patterns handle: "Key" [...] or 'Key' [...] (no colon)
        # Also handle: "Key" (...) [...] or 'Key' (...) [...] (with parenthetical comment, no colon)
        # Pattern 1: double-quoted key, optional comment, space+bracket (no colon)
        kv_match = re.match(r""""([^"]+)"(?:\s*\([^)]*\))?(?!\s*:)\s*\[([^\]]+)\]""", line)
        if kv_match:
            key = kv_match.group(1)
            values_str = kv_match.group(2)
            values = [v.strip().strip("'\"").strip('"') for v in values_str.split(",")]
            values = [v for v in values if v]
            if key and values:
                mapping[key] = values
                continue

        # Pattern 2: single-quoted key, optional comment, space+bracket (no colon)
        kv_match = re.match(r"""'([^']+)'(?:\s*\([^)]*\))?(?!\s*:)\s*\[([^\]]+)\]""", line)
        if kv_match:
            key = kv_match.group(1)
            values_str = kv_match.group(2)
            values = [v.strip().strip("'\"").strip('"') for v in values_str.split(",")]
            values = [v for v in values if v]
            if key and values:
                mapping[key] = values
                continue

        # Pattern 3: double-quoted key WITH parenthetical comment AND colon (Mistral specific)
        kv_match = re.match(r""""([^"]+)"(?:\s*\([^)]*\)):\s*\[([^\]]+)\]""", line)
        if kv_match:
            key = kv_match.group(1)
            values_str = kv_match.group(2)
            values = [v.strip().strip("'\"").strip('"') for v in values_str.split(",")]
            values = [v for v in values if v]
            if key and values:
                mapping[key] = values
                continue

        # Pattern 4: single-quoted key WITH parenthetical comment AND colon (Mistral specific)
        kv_match = re.match(r"""'([^']+)'(?:\s*\([^)]*\)):\s*\[([^\]]+)\]""", line)
        if kv_match:
            key = kv_match.group(1)
            values_str = kv_match.group(2)
            values = [v.strip().strip("'\"").strip('"') for v in values_str.split(",")]
            values = [v for v in values if v]
            if key and values:
                mapping[key] = values
                continue

        # Pattern 5: double outer quotes, single-quoted key, parenthetical comment (Mistral)
        # e.g., "'jun food dominance index' (JFD*)": ["value1", "value2"]
        # The key is: 'jun food dominance index' wrapped in double quotes
        _p5 = (
            '"' + "'" + "([^" + "'" + "]+)" + "'" + r"(?:\s*\([^)]*\))" + r":\s*\[" + r"([^\]]+)\]"
        )
        kv_match = re.match(_p5, line)
        if kv_match:
            key = "'" + kv_match.group(1) + "'"
            values_str = kv_match.group(2)
            values = [v.strip().strip("'\"").strip('"') for v in values_str.split(",")]
            values = [v for v in values if v]
            if key and values:
                mapping[key] = values

    if mapping:
        return mapping

    first_entry = _recover_first_quoted_children_entry(block)
    if first_entry is not None and not _children_mapping_needs_more_recovery(
        text=block,
        parsed=first_entry,
    ):
        return first_entry
    return None


def _recover_first_quoted_children_entry(text: str) -> dict[str, list[str]] | None:
    patterns = (
        r"""'([^']+)'\s*:\s*\[(.*?)\]""",
        r""""([^"]+)"\s*:\s*\[(.*?)\]""",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.DOTALL)
        if match is None:
            continue
        key = match.group(1).strip()
        values_result = _scan_lenient_quoted_list(f"[{match.group(2)}]", 0)
        if values_result is None:
            continue
        values, _ = values_result
        if key and values:
            return {key: values}
    return None


def _recover_single_key_children_values(text: str) -> dict[str, list[str]] | None:
    match = re.search(r"""['"]([^'"]+)['"]\s*:\s*\[""", text, flags=re.DOTALL)
    if match is None:
        return None
    key = match.group(1).strip()
    if not key:
        return None
    values_text = text[match.end() :]
    values = [
        item.strip() for item in re.findall(r"""['"]([^'"]+)['"]""", values_text, flags=re.DOTALL)
    ]
    values = [value for value in values if value]
    if not values:
        return None
    return {key: values}


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
    print(f"DEBUG: needs_more_recovery called with len={len(parsed)}")
    if len(parsed) != 1:
        return False
    values = next(iter(parsed.values()))
    if any(
        value.startswith(('"', "'"))
        or value.endswith(('"', "'"))
        or "[" in value
        or "]" in value
        or "\n" in value
        for value in values
    ):
        return True
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
    while True:
        before = result
        result = re.sub(
            r"\[([^'\"]*'[^'\"]*'[^'\"]*)'?\s*\]",
            lambda m: _extract_quoted_strings_from_bracket(m.group(1)),
            result,
        )
        if result == before:
            break
    return result


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
    candidates: list[str] = [stripped]
    if stripped.count("[") > stripped.count("]"):
        candidates.append(f"{stripped}]")
    if stripped.count("{") > stripped.count("}"):
        candidates.append(f"{stripped}}}")
    if stripped.count("[") > stripped.count("]") and stripped.count("{") > stripped.count("}"):
        candidates.append(f"{stripped}]}}")
    # Also handle case where list is open but outer brace is closed: {Key: [val1, val2}
    if stripped.endswith("}") and stripped.count("[") > stripped.count("]"):
        candidates.append(f"{stripped[:-1]}]}}")
    # Handle trailing comma before ] or }: ['val1', 'val2', ]} -> ['val1', 'val2']}
    # Pattern: comma followed by close brace and close bracket: '}, ]'
    trailing_comma_fixed = re.sub(r",\s*\}\s*\]", "]", stripped)
    if trailing_comma_fixed != stripped:
        candidates.append(trailing_comma_fixed)
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _recover_children_mapping_from_lines(text: str) -> dict[str, list[str]] | None:
    from llm_conceptual_modeling.common.hf_transformers._label_list import (
        _extract_outer_block,
        _extract_quoted_line_value,
    )

    block = _extract_outer_block(text=text, opener="{", closer="}")
    candidate_text = block if block is not None else text
    sanitized_text = _strip_mapping_comments(candidate_text)
    lines = [line.strip() for line in sanitized_text.splitlines() if line.strip()]
    if not lines:
        return None
    mapping: dict[str, list[str]] = {}
    key: str | None = None
    values: list[str] = []
    in_children = False

    def flush_current_entry() -> None:
        nonlocal key, values, in_children
        if key is not None:
            mapping[key] = values
        key = None
        values = []
        in_children = False

    for line in lines:
        if line in {"{", "}"}:
            if line == "}":
                flush_current_entry()
            continue
        if ":" in line:
            line_key, line_value = line.split(":", 1)
            candidate_key = _extract_quoted_line_value(line_key)
            if candidate_key is not None:
                flush_current_entry()
                key = candidate_key
                if "[" not in line_value:
                    continue
                in_children = True
                value = _extract_quoted_line_value(line_value.split("[", 1)[1])
                if value is not None:
                    values.append(value)
                if "]" in line_value.split("[", 1)[1]:
                    flush_current_entry()
                continue
        if key is None:
            continue
        if not in_children:
            continue
        if line.startswith("]") or line.startswith("}"):
            flush_current_entry()
            continue
        value = _extract_quoted_line_value(line)
        if value is None:
            continue
        values.append(value)
    flush_current_entry()
    if not mapping:
        return None
    if _children_mapping_needs_more_recovery(text=sanitized_text, parsed=mapping):
        return None
    return mapping
