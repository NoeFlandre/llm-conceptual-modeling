from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from typing import cast


def _skip_inline_mapping_separators(text: str, index: int) -> int:
    while index < len(text) and text[index] in {" ", "\t", "\n", "\r"}:
        index += 1
    return index


def _scan_lenient_quoted_string(text: str, start_index: int) -> tuple[str, int] | None:
    if start_index >= len(text) or text[start_index] not in {"'", '"'}:
        return None
    opening_quote = text[start_index]
    delimiter_chars = {",", "]", "}", ":", ")"}
    index = start_index + 1
    while index < len(text):
        current_char = text[index]
        if current_char in {"'", '"'}:
            next_index = _skip_inline_mapping_separators(text, index + 1)
            if next_index >= len(text) or text[next_index] in delimiter_chars:
                if current_char != opening_quote:
                    matching_quote_index = text.find(opening_quote, index + 1)
                    if matching_quote_index != -1:
                        index += 1
                        continue
                value = text[start_index + 1 : index].strip()
                return value, index + 1
            if next_index < len(text) and text[next_index] in {"'", '"'}:
                next_next_index = _skip_inline_mapping_separators(text, next_index + 1)
                if next_next_index < len(text) and text[next_next_index] in delimiter_chars:
                    if current_char == opening_quote:
                        value = text[start_index + 1 : index].strip()
                        return value, next_index + 1
            if (
                current_char == opening_quote
                and index > start_index + 1
                and index + 1 < len(text)
                and text[index - 1].isalnum()
                and text[index + 1].isalnum()
            ):
                index += 1
                continue
        index += 1
    return None


def _normalize_label_list_payload(parsed: object) -> list[str] | None:
    if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
        return cast(list[str], parsed)
    if not isinstance(parsed, Mapping):
        return None
    parsed_mapping = cast(Mapping[str, object], parsed)
    labels = parsed_mapping.get("labels")
    if isinstance(labels, list) and all(isinstance(item, str) for item in labels):
        string_labels = cast(list[str], labels)
        if len(labels) == 1:
            packed_labels = _split_packed_label_string(string_labels[0])
            if packed_labels is not None:
                return packed_labels
        return string_labels
    if isinstance(labels, str):
        return _split_packed_label_string(labels)
    return None


def _split_packed_label_string(text: str) -> list[str] | None:
    if "', '" not in text and '", "' not in text:
        stripped = text.strip().strip("'\"")
        return [stripped] if stripped else None
    parts = re.split(r"""['"]\s*,\s*['"]""", text.strip())
    cleaned_parts = [part.strip().strip("'\"") for part in parts if part.strip().strip("'\"")]
    if not cleaned_parts:
        return None
    return cleaned_parts


def _recover_label_list_from_lines(text: str) -> list[str] | None:
    block = _extract_outer_block(text=text, opener="[", closer="]")
    if block is None:
        block = _recover_truncated_list_block(text)
        if block is None:
            from_segments = _recover_label_list_from_segments(text)
            if from_segments is not None:
                return from_segments
            return _recover_label_list_from_quoted_candidates(text)
    labels: list[str] = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or line in {"[", "]"}:
            continue
        label = _extract_quoted_line_value(line)
        if (
            label is not None
            and "\n" not in block
            and sum(line.count(quote) for quote in ("'", '"')) > 2
        ):
            from_segments = _recover_label_list_from_segments(text)
            if from_segments is not None:
                return from_segments
        if label is None:
            return _recover_label_list_from_segments(text)
        labels.append(label)
    if not labels:
        from_segments = _recover_label_list_from_segments(text)
        if from_segments is not None:
            return from_segments
        return _recover_label_list_from_quoted_candidates(text)
    if len(labels) == 1 and _looks_like_packed_label(labels[0]):
        from_segments = _recover_label_list_from_segments(text)
        if from_segments is not None:
            return from_segments
    return labels


def _recover_truncated_list_block(text: str) -> str | None:
    stripped = text.strip()
    if not stripped.startswith("[") or stripped.endswith("]"):
        return None
    candidate = f"{stripped}]"
    try:
        parsed = ast.literal_eval(candidate)
    except (ValueError, SyntaxError):
        return None
    if (
        not isinstance(parsed, list)
        or not parsed
        or not all(isinstance(item, str) for item in parsed)
    ):
        return None
    return candidate


def _recover_label_list_from_segments(text: str) -> list[str] | None:
    stripped = text.strip()
    if not stripped.startswith("["):
        return None
    body = stripped[1:].strip()
    if not body:
        return None
    segments = [segment.strip() for segment in body.split(",")]
    labels: list[str] = []
    for segment in segments:
        cleaned = segment.strip().strip("[]").strip().strip("'\"")
        if not cleaned:
            continue
        if any(marker in cleaned for marker in ("```", "{", "}", ":", "assistant")):
            return None
        labels.append(cleaned)
    if len(labels) < 2:
        return labels if len(labels) == 1 else None
    return labels


def _recover_label_list_from_quoted_candidates(text: str) -> list[str] | None:
    quoted_items = re.findall(r"""['"]([^'"]+)['"]""", text)
    if len(quoted_items) < 3:
        return None
    labels: list[str] = []
    seen: set[str] = set()
    for item in quoted_items:
        candidate = item.strip()
        if not _looks_like_recoverable_label(candidate):
            continue
        normalized_candidate = candidate.casefold()
        if normalized_candidate in seen:
            continue
        seen.add(normalized_candidate)
        labels.append(candidate)
    if len(labels) < 3:
        return None
    return labels[:5]


def _recover_bare_comma_separated_label_list(text: str) -> list[str] | None:
    stripped = text.strip()
    if not stripped or stripped.startswith("["):
        return None
    if any(marker in stripped for marker in "{}:`"):
        return None
    parts = [part.strip().strip("'\"") for part in stripped.split(",")]
    labels = [part for part in parts if part and _looks_like_recoverable_label(part)]
    if len(labels) < 2:
        return None
    return labels[:5]


def _recover_quoted_label_list_with_comments(text: str) -> list[str] | None:
    if "#" not in text and "**Note:**" not in text and "**note:**" not in text:
        return None
    quoted_items = re.findall(r"""['"]([^'"]+)['"]""", text)
    if len(quoted_items) < 2:
        return None
    labels: list[str] = []
    seen: set[str] = set()
    for item in quoted_items:
        candidate = item.strip()
        if not _looks_like_recoverable_label(candidate):
            continue
        normalized = candidate.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        labels.append(candidate)
    if len(labels) < 2:
        return None
    return labels[:5]


def _recover_single_bare_label(text: str) -> list[str] | None:
    stripped = text.strip()
    if not stripped:
        return None
    if any(marker in stripped for marker in "[]{}:,`\n"):
        return None
    if stripped.casefold().startswith("assistant"):
        return None
    candidate = stripped.strip("'\"").strip()
    if not _looks_like_recoverable_label(candidate):
        return None
    return [candidate]


def _looks_like_recoverable_label(text: str) -> bool:
    if len(text) <= 1:
        return False
    return bool(re.search(r"[A-Za-z]{2,}", text))


def _extract_outer_block(*, text: str, opener: str, closer: str) -> str | None:
    start = text.find(opener)
    end = text.rfind(closer)
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _extract_first_balanced_block(text: str, *, opener: str, closer: str) -> str | None:
    start = text.find(opener)
    if start == -1:
        return None
    depth = 0
    for index in range(start, len(text)):
        if text[index] == opener:
            depth += 1
        elif text[index] == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _looks_like_packed_label(text: str) -> bool:
    return "', '" in text or '", "' in text


def _extract_quoted_line_value(text: str) -> str | None:
    stripped = text.strip().rstrip(",")
    first_quote_positions = [
        position for position in (stripped.find("'"), stripped.find('"')) if position != -1
    ]
    if not first_quote_positions:
        return None
    start = min(first_quote_positions)
    parsed = _scan_lenient_quoted_string(stripped, start)
    if parsed is not None:
        value, _ = parsed
        cleaned_value = value.strip()
        if (
            len(cleaned_value) >= 2
            and cleaned_value[0] in {"'", '"'}
            and cleaned_value[-1] == cleaned_value[0]
        ):
            cleaned_value = cleaned_value[1:-1].strip()
        return cleaned_value
    return None
