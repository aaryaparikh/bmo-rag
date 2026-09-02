"""Load the normalized Docling chunk corpus used by indexing and evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CorpusChunk:
    chunk_id: str
    source_id: str
    chunk_index: int
    text: str
    headings: tuple[str, ...]
    pages: tuple[int, ...]
    source_url: str | None
    origin_filename: str | None

    @property
    def embedding_text(self) -> str:
        """Include headings consistently because they carry useful retrieval context."""
        if not self.headings:
            return self.text
        return f"{' > '.join(self.headings)}\n\n{self.text}"

    def payload(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "headings": list(self.headings),
            "pages": list(self.pages),
            "source_url": self.source_url,
            "origin_filename": self.origin_filename,
        }


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def stable_chunk_id(source_id: str, text: str) -> str:
    digest = hashlib.sha256(f"{source_id}\0{normalize_text(text)}".encode()).hexdigest()[:20]
    return f"bmo-{digest}"


def _pages(meta: dict[str, Any]) -> tuple[int, ...]:
    found: set[int] = set()
    for item in meta.get("doc_items", []):
        for provenance in item.get("prov", []):
            page = provenance.get("page_no")
            if isinstance(page, int):
                found.add(page)
    return tuple(sorted(found))


def load_corpus(chunk_dir: Path) -> list[CorpusChunk]:
    chunks: list[CorpusChunk] = []
    for path in sorted(chunk_dir.glob("*.chunks.jsonl")):
        source_id = path.name.removesuffix(".chunks.jsonl")
        for chunk_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            raw = json.loads(line)
            meta = raw.get("meta", {})
            text = normalize_text(raw["text"])
            headings = tuple(
                heading
                for value in meta.get("headings", [])
                if (heading := normalize_text(str(value)))
            )
            chunks.append(
                CorpusChunk(
                    chunk_id=stable_chunk_id(source_id, text),
                    source_id=source_id,
                    chunk_index=chunk_index,
                    text=text,
                    headings=headings,
                    pages=_pages(meta),
                    source_url=meta.get("source_url"),
                    origin_filename=(meta.get("origin") or {}).get("filename"),
                )
            )
    if not chunks:
        raise ValueError(f"No *.chunks.jsonl files found in {chunk_dir}")
    return chunks


def group_by_source(chunks: Iterable[CorpusChunk]) -> dict[str, list[CorpusChunk]]:
    grouped: dict[str, list[CorpusChunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.source_id, []).append(chunk)
    return grouped
