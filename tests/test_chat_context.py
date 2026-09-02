from __future__ import annotations

from bmo_rag.retrieval.context import expand_points, pack_context


def point(index: int, *, heading: str = "Capital") -> dict:
    return {
        "id": str(index),
        "score": 1.0,
        "payload": {
            "chunk_id": f"chunk-{index}",
            "source_id": "report",
            "chunk_index": index,
            "origin_filename": "report.pdf",
            "pages": [index + 1],
            "headings": [heading],
            "text": f"Passage {index}.",
        },
    }


class FakeStore:
    def related_points(
        self,
        name: str,
        *,
        source_id: str,
        chunk_indices: list[int] | None = None,
        heading: str | None = None,
        limit: int = 32,
    ) -> list[dict]:
        values = [point(index) for index in range(5)]
        if chunk_indices is not None:
            values = [item for item in values if item["payload"]["chunk_index"] in chunk_indices]
        if heading is not None:
            values = [item for item in values if heading in item["payload"]["headings"]]
        return values[:limit]


def test_broad_query_preserves_seeds_and_adds_same_section_neighbors() -> None:
    seeds = [point(2), point(4)]
    result = expand_points(
        "Explain the change in capital",
        seeds,
        store=FakeStore(),
        collection="chunks",
        max_chunks=5,
    )

    assert [item["payload"]["chunk_id"] for item in result[:2]] == ["chunk-2", "chunk-4"]
    assert all(item["payload"]["headings"] == ["Capital"] for item in result)
    assert len(result) == 5


def test_context_includes_citation_heading_and_source_labels() -> None:
    packed = pack_context([point(2)], max_chars=1000)

    assert "[S1] Source: report.pdf, p. 3" in packed.text
    assert "Section: Capital" in packed.text
    assert packed.sources[0].chunk_id == "chunk-2"
    assert packed.truncated is False
