from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any


def normalize_chunk_sizes(
    chunks: list[dict[str, Any]],
    *,
    min_chars: int,
    max_chars: int,
    overlap_chars: int,
    preserve_sections: bool = False,
) -> list[dict[str, Any]]:
    """Merge very small section-local chunks and split chunks over the max size."""
    validate_chunk_size_options(
        min_chunk_chars=min_chars,
        max_chunk_chars=max_chars,
        chunk_overlap_chars=overlap_chars,
    )
    merged = _merge_small_chunks(chunks, min_chars=min_chars)
    if not preserve_sections:
        merged = _merge_remaining_small_chunks(merged, min_chars=min_chars)
    normalized: list[dict[str, Any]] = []
    for chunk in merged:
        text = _chunk_text(chunk)
        if len(text) > max_chars:
            normalized.extend(
                _split_large_chunk(chunk, max_chars=max_chars, overlap_chars=overlap_chars)
            )
        elif text.strip():
            normalized.append(chunk)
    return _merge_remaining_small_chunks_with_max(
        normalized,
        min_chars=min_chars,
        max_chars=max_chars,
        same_section_only=preserve_sections,
    )


def _merge_small_chunks(chunks: list[dict[str, Any]], *, min_chars: int) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None

    for chunk in chunks:
        if not _chunk_text(chunk).strip():
            continue
        if pending is None:
            pending = deepcopy(chunk)
            continue

        if len(_chunk_text(pending)) < min_chars and _same_section(pending, chunk):
            pending = _merge_chunk_pair(pending, chunk)
            continue

        merged.append(pending)
        pending = deepcopy(chunk)

    if pending is not None:
        merged.append(pending)
    return merged


def _merge_chunk_pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(left)
    left_text = _chunk_text(left).strip()
    right_text = _chunk_text(right).strip()
    merged["text"] = "\n\n".join(text for text in [left_text, right_text] if text)
    merged["meta"] = _merge_meta(left.get("meta"), right.get("meta"))
    strategy = "merged_small" if _same_section(left, right) else "merged_small_cross_section"
    _set_normalization_metadata(
        merged,
        strategy=strategy,
        source_chunk_count=_normalization_source_count(left) + _normalization_source_count(right),
    )
    return merged


def _merge_meta(left: Any, right: Any) -> Any:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return left if left is not None else right

    merged = deepcopy(left)
    if "doc_items" in left or "doc_items" in right:
        merged["doc_items"] = _dedupe_jsonish_items(
            [*left.get("doc_items", []), *right.get("doc_items", [])]
        )
    if "captions" in left or "captions" in right:
        merged["captions"] = _dedupe_jsonish_items(
            [*left.get("captions", []), *right.get("captions", [])]
        )
    if "headings" in left or "headings" in right:
        merged["headings"] = _dedupe_jsonish_items(
            [*left.get("headings", []), *right.get("headings", [])]
        )
    return merged


def _merge_remaining_small_chunks(
    chunks: list[dict[str, Any]],
    *,
    min_chars: int,
) -> list[dict[str, Any]]:
    if len(chunks) <= 1:
        return chunks

    merged: list[dict[str, Any]] = []
    index = 0
    while index < len(chunks):
        chunk = chunks[index]
        if len(_chunk_text(chunk)) >= min_chars:
            merged.append(chunk)
            index += 1
            continue

        if index + 1 < len(chunks):
            merged.append(_merge_chunk_pair(chunk, chunks[index + 1]))
            index += 2
            continue

        if merged:
            merged[-1] = _merge_chunk_pair(merged[-1], chunk)
        else:
            merged.append(chunk)
        index += 1

    return merged


def _merge_remaining_small_chunks_with_max(
    chunks: list[dict[str, Any]],
    *,
    min_chars: int,
    max_chars: int,
    same_section_only: bool = False,
) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for chunk in chunks:
        if len(_chunk_text(chunk)) >= min_chars:
            compacted.append(chunk)
            continue

        if (
            compacted
            and _combined_text_len(compacted[-1], chunk) <= max_chars
            and (not same_section_only or _same_section(compacted[-1], chunk))
        ):
            compacted[-1] = _merge_chunk_pair(compacted[-1], chunk)
            continue

        compacted.append(chunk)

    index = 0
    while index < len(compacted) - 1:
        if (
            len(_chunk_text(compacted[index])) < min_chars
            and _combined_text_len(compacted[index], compacted[index + 1]) <= max_chars
            and (
                not same_section_only
                or _same_section(compacted[index], compacted[index + 1])
            )
        ):
            compacted[index] = _merge_chunk_pair(compacted[index], compacted[index + 1])
            compacted.pop(index + 1)
            continue
        index += 1

    return compacted


def _combined_text_len(left: dict[str, Any], right: dict[str, Any]) -> int:
    return len(_chunk_text(left).strip()) + len(_chunk_text(right).strip()) + 2


def _split_large_chunk(
    chunk: dict[str, Any],
    *,
    max_chars: int,
    overlap_chars: int,
) -> list[dict[str, Any]]:
    text = _chunk_text(chunk)
    segments = _split_text(text, max_chars=max_chars, overlap_chars=overlap_chars)
    split_chunks: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        split_chunk = deepcopy(chunk)
        split_chunk["text"] = segment
        _set_normalization_metadata(
            split_chunk,
            strategy="split_large",
            split_index=index,
            split_count=len(segments),
            parent_char_count=len(text),
        )
        split_chunks.append(split_chunk)
    return split_chunks


def _split_text(text: str, *, max_chars: int, overlap_chars: int) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    segments: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                segments.append(current)
                current = ""
            segments.extend(_split_long_paragraph(paragraph, max_chars=max_chars, overlap_chars=overlap_chars))
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                segments.append(current)
            current = paragraph

    if current:
        segments.append(current)
    return segments


def _split_long_paragraph(paragraph: str, *, max_chars: int, overlap_chars: int) -> list[str]:
    words = paragraph.split()
    segments: list[str] = []
    current = ""

    for word in words:
        if len(word) > max_chars:
            if current:
                segments.append(current)
                current = ""
            segments.extend(
                word[start : start + max_chars]
                for start in range(0, len(word), max_chars - overlap_chars)
            )
            continue

        candidate = f"{current} {word}" if current else word
        if len(candidate) <= max_chars:
            current = candidate
            continue

        segments.append(current)
        overlap = _tail_overlap(current, overlap_chars)
        current = f"{overlap} {word}".strip() if overlap else word

    if current:
        segments.append(current)
    return segments


def _tail_overlap(text: str, overlap_chars: int) -> str:
    if overlap_chars <= 0 or len(text) <= overlap_chars:
        return ""
    tail = text[-overlap_chars:]
    first_space = tail.find(" ")
    return tail[first_space + 1 :].strip() if first_space >= 0 else tail.strip()


def _same_section(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return _headings(left) == _headings(right)


def _headings(chunk: dict[str, Any]) -> tuple[str, ...]:
    meta = chunk.get("meta")
    if not isinstance(meta, dict):
        return ()
    headings = meta.get("headings") or []
    return tuple(str(heading) for heading in headings)


def _chunk_text(chunk: dict[str, Any]) -> str:
    text = chunk.get("text", "")
    return text if isinstance(text, str) else str(text)


def _dedupe_jsonish_items(items: list[Any]) -> list[Any]:
    seen: set[str] = set()
    deduped: list[Any] = []
    for item in items:
        key = json.dumps(item, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def _normalization_source_count(chunk: dict[str, Any]) -> int:
    meta = chunk.get("meta")
    if not isinstance(meta, dict):
        return 1
    normalization = meta.get("chunk_normalization")
    if not isinstance(normalization, dict):
        return 1
    value = normalization.get("source_chunk_count", 1)
    return value if isinstance(value, int) else 1


def _set_normalization_metadata(chunk: dict[str, Any], **values: Any) -> None:
    meta = chunk.setdefault("meta", {})
    if not isinstance(meta, dict):
        chunk["meta"] = {"chunk_normalization": values}
        return
    existing = meta.get("chunk_normalization", {})
    if not isinstance(existing, dict):
        existing = {}
    meta["chunk_normalization"] = {**existing, **values}


def validate_chunk_size_options(
    *,
    min_chunk_chars: int,
    max_chunk_chars: int,
    chunk_overlap_chars: int,
) -> None:
    if min_chunk_chars < 0:
        raise ValueError("min_chunk_chars must be non-negative")
    if max_chunk_chars <= 0:
        raise ValueError("max_chunk_chars must be greater than 0")
    if min_chunk_chars > max_chunk_chars:
        raise ValueError("min_chunk_chars must be less than or equal to max_chunk_chars")
    if chunk_overlap_chars < 0 or chunk_overlap_chars >= max_chunk_chars:
        raise ValueError("chunk_overlap_chars must be non-negative and smaller than max_chunk_chars")
