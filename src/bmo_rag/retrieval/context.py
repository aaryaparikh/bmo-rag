"""Selective neighbor/section expansion and citation-aware context packing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from bmo_rag.indexing.qdrant_store import QdrantStore
from bmo_rag.retrieval.semantic import citation

BROAD_QUERY = re.compile(
    r"\b(why|explain|summari[sz]e|compare|comparison|trend|reasons?|drivers?|"
    r"list|all|breakdown|how (?:did|has|does)|change[sd]?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PackedSource:
    label: str
    chunk_id: str
    citation: str
    headings: tuple[str, ...]
    text: str
    role: str


@dataclass(frozen=True)
class PackedContext:
    text: str
    sources: tuple[PackedSource, ...]
    truncated: bool


def expand_points(
    question: str,
    seeds: list[dict[str, Any]],
    *,
    store: QdrantStore,
    collection: str,
    neighbor_window: int = 1,
    section_window: int = 2,
    max_chunks: int = 14,
) -> list[dict[str, Any]]:
    """Expand reranked seeds without indiscriminately adding every adjacent chunk."""
    broad = bool(BROAD_QUERY.search(question))
    expanded: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(point: dict[str, Any], role: str, seed_rank: int) -> None:
        payload = point.get("payload") or {}
        chunk_id = str(payload.get("chunk_id") or point.get("id"))
        if chunk_id in seen or len(expanded) >= max_chunks:
            return
        seen.add(chunk_id)
        expanded.append({**point, "context_role": role, "seed_rank": seed_rank})

    # Preserve every reranked seed before spending the remaining budget on expansion.
    for seed_rank, seed in enumerate(seeds, start=1):
        add(seed, "seed", seed_rank)

    for seed_rank, seed in enumerate(seeds, start=1):
        payload = seed.get("payload") or {}
        source_id = payload.get("source_id")
        chunk_index = payload.get("chunk_index")
        headings = tuple(str(value) for value in payload.get("headings") or [])
        text = str(payload.get("text") or "").strip()
        if not isinstance(source_id, str) or not isinstance(chunk_index, int):
            continue

        should_expand = broad or _looks_boundary_limited(text)
        related: list[dict[str, Any]] = []
        if should_expand and neighbor_window:
            indices = [
                index
                for index in range(chunk_index - neighbor_window, chunk_index + neighbor_window + 1)
                if index >= 0 and index != chunk_index
            ]
            related.extend(
                store.related_points(
                    collection,
                    source_id=source_id,
                    chunk_indices=indices,
                    limit=len(indices),
                )
            )
        if broad and headings and section_window > neighbor_window:
            section = store.related_points(
                collection,
                source_id=source_id,
                heading=headings[-1],
                limit=32,
            )
            related.extend(
                point
                for point in section
                if abs(int((point.get("payload") or {}).get("chunk_index", -999)) - chunk_index)
                <= section_window
            )

        valid_related = [
            point
            for point in related
            if tuple(str(value) for value in (point.get("payload") or {}).get("headings") or [])
            == headings
        ]
        for point in sorted(
            valid_related,
            key=lambda item: (
                abs(int((item.get("payload") or {}).get("chunk_index", 0)) - chunk_index),
                int((item.get("payload") or {}).get("chunk_index", 0)),
            ),
        ):
            add(point, "expanded", seed_rank)
    return expanded


def pack_context(points: list[dict[str, Any]], *, max_chars: int = 32000) -> PackedContext:
    """Format source metadata and text under a deterministic context budget."""
    blocks: list[str] = []
    sources: list[PackedSource] = []
    used = 0
    truncated = False
    for point in points:
        payload = point.get("payload") or {}
        label = f"S{len(sources) + 1}"
        headings = tuple(str(value) for value in payload.get("headings") or [])
        heading_text = " > ".join(headings) or "Unsectioned"
        source_citation = citation(payload)
        body = str(payload.get("text") or "").strip()
        block = f"[{label}] Source: {source_citation}\nSection: {heading_text}\n{body}"
        if used + len(block) > max_chars:
            truncated = True
            continue
        blocks.append(block)
        used += len(block) + 2
        sources.append(
            PackedSource(
                label=label,
                chunk_id=str(payload.get("chunk_id") or point.get("id") or "unknown"),
                citation=source_citation,
                headings=headings,
                text=body,
                role=str(point.get("context_role") or "seed"),
            )
        )
    return PackedContext(text="\n\n".join(blocks), sources=tuple(sources), truncated=truncated)


def _looks_boundary_limited(text: str) -> bool:
    if not text:
        return False
    begins_mid_sentence = text[0].islower()
    ends_mid_sentence = text[-1] not in ".!?)]}"
    table_like = text.count("|") >= 4 or bool(
        re.search(r"\btable\b", text[:100], re.IGNORECASE)
    )
    return begins_mid_sentence or ends_mid_sentence or table_like
