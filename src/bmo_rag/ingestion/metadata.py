from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

from bmo_rag.ingestion.sources import RawSource

CONFIDENCE_KEYS = {"confidence", "conf", "confidence_score"}


def confidence_summary(
    *,
    document_dict: dict[str, Any],
    conversion_status: str,
    chunk_count: int,
    failed: bool,
) -> dict[str, Any]:
    item_scores = list(_find_confidence_scores(document_dict))
    conversion_score = 0.0 if failed else (1.0 if conversion_status == "success" else 0.5)
    item_average = round(sum(item_scores) / len(item_scores), 4) if item_scores else None
    extraction_score = item_average if item_average is not None else conversion_score
    chunking_score = 0.0 if failed else (1.0 if chunk_count > 0 else 0.5)

    return {
        "conversion": conversion_score,
        "extraction": extraction_score,
        "chunking": chunking_score,
        "docling_item_scores": {
            "count": len(item_scores),
            "average": item_average,
            "minimum": round(min(item_scores), 4) if item_scores else None,
            "maximum": round(max(item_scores), 4) if item_scores else None,
        },
    }


def _find_confidence_scores(value: Any) -> Iterable[float]:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in CONFIDENCE_KEYS and isinstance(nested, int | float) and not isinstance(nested, bool):
                yield float(nested)
            else:
                yield from _find_confidence_scores(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _find_confidence_scores(item)


def build_metadata(
    source: RawSource,
    status: str,
    confidence: dict[str, Any],
    started_at: float,
    chunk_count: int,
    *,
    device: str = "auto",
    num_threads: int = 4,
    do_ocr: bool = False,
    force_backend_text: bool = True,
    do_table_structure: bool = True,
    min_chunk_chars: int = 300,
    max_chunk_chars: int = 1500,
    chunk_overlap_chars: int = 150,
    error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source": {
            "location": str(source.location),
            "type": source.source_type,
            "id": source.source_id,
            "document_kind": source.document_kind,
            "display_name": source.display_name,
        },
        "status": status,
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "chunk_count": chunk_count,
        "accelerator": {
            "device": device,
            "num_threads": num_threads,
        },
        "docling_options": {
            "do_ocr": do_ocr,
            "force_backend_text": force_backend_text,
            "do_table_structure": do_table_structure,
        },
        "chunk_size_options": {
            "min_chars": min_chunk_chars,
            "max_chars": max_chunk_chars,
            "overlap_chars": chunk_overlap_chars,
        },
        "confidence": confidence,
    }
    if error:
        payload["error"] = error
    return payload

