from __future__ import annotations

import hashlib
import re
import unicodedata
from copy import deepcopy
from typing import Any

MAX_RECORDED_DUPLICATE_OCCURRENCES = 25
CHUNK_ID_PREFIX = "bmo-"
CHUNK_ID_DIGEST_LENGTH = 20


def add_native_chunk_ids(
    chunks: list[dict[str, Any]],
    *,
    source_id: str,
) -> list[dict[str, Any]]:
    """Add stable, content-derived IDs to chunks from one source."""
    identified: list[dict[str, Any]] = []
    for original in chunks:
        chunk = deepcopy(original)
        normalized_text = re.sub(r"\s+", " ", _chunk_text(chunk)).strip()
        digest = hashlib.sha256(f"{source_id}\0{normalized_text}".encode()).hexdigest()[
            :CHUNK_ID_DIGEST_LENGTH
        ]
        chunk["chunk_id"] = f"{CHUNK_ID_PREFIX}{digest}"
        identified.append(chunk)
    return identified


def deduplicate_chunks(
    chunks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove normalized exact duplicates while recording compact occurrence metadata."""
    retained: list[dict[str, Any]] = []
    retained_by_key: dict[str, dict[str, Any]] = {}
    duplicates_removed = 0

    for source_index, original in enumerate(chunks):
        chunk = deepcopy(original)
        key = normalized_text_key(_chunk_text(chunk))
        if not key:
            continue

        existing = retained_by_key.get(key)
        if existing is None:
            retained_by_key[key] = chunk
            retained.append(chunk)
            continue

        duplicates_removed += 1
        meta = existing.setdefault("meta", {})
        if not isinstance(meta, dict):
            meta = {}
            existing["meta"] = meta
        audit = meta.setdefault(
            "deduplication",
            {
                "strategy": "normalized_exact",
                "duplicate_count": 0,
                "occurrences": [],
                "occurrences_truncated": 0,
            },
        )
        audit["duplicate_count"] += 1
        occurrence = _duplicate_occurrence(original, source_index=source_index)
        if len(audit["occurrences"]) < MAX_RECORDED_DUPLICATE_OCCURRENCES:
            audit["occurrences"].append(occurrence)
        else:
            audit["occurrences_truncated"] += 1

    return retained, {
        "enabled": True,
        "strategy": "normalized_exact",
        "input_count": len(chunks),
        "output_count": len(retained),
        "duplicates_removed": duplicates_removed,
    }


def normalized_text_key(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", value).strip()


def _duplicate_occurrence(chunk: dict[str, Any], *, source_index: int) -> dict[str, Any]:
    meta = chunk.get("meta") if isinstance(chunk.get("meta"), dict) else {}
    pages = sorted(
        {
            provenance["page_no"]
            for item in meta.get("doc_items", [])
            if isinstance(item, dict)
            for provenance in item.get("prov", [])
            if isinstance(provenance, dict) and isinstance(provenance.get("page_no"), int)
        }
    )
    occurrence: dict[str, Any] = {"source_chunk_index": source_index}
    if pages:
        occurrence["pages"] = pages
    headings = meta.get("headings")
    if isinstance(headings, list) and headings:
        occurrence["headings"] = headings
    section_index = meta.get("section_index")
    if isinstance(section_index, int):
        occurrence["section_index"] = section_index
    return occurrence


def _chunk_text(chunk: dict[str, Any]) -> str:
    value = chunk.get("text", "")
    return value if isinstance(value, str) else str(value)
